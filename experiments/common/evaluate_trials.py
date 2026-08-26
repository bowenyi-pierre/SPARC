"""
Evaluates all models stored in the JSON score files produced by extract_scores.py.

Input layout
------------
  --scored_dir   directory with 4 JSON files:
                   easy_positives.json  hard_positives.json
                   easy_negatives.json  hard_negatives.json
                 Keys are trial_ids; values are {model: score} dicts.

  --trial_dir    directory with the matching 4 CSV files (labels + metadata).

  --task         speaker | gender | age | accent   (for plot titles only)

Output
------
  roc_combined.png          ROC on all trials, all models
  roc_easy.png              easy_pos + easy_neg
  roc_hard.png              hard_pos + hard_neg
  roc_same_topic.png        easy_pos + hard_neg  (same podcast)
  roc_diff_topic.png        hard_pos + easy_neg  (different podcast)
  summary.json              EER, AUC, TPR@FPR, epsilon per model per view

Usage
-----
    python evaluate.py \\
        --scored_dir scored/speaker_50_L5/ \\
        --trial_dir  trials/50_L5/speaker/ \\
        --task       speaker \\
        --output_dir eval/speaker_50_L5/
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve



TRIAL_FILES = [
    "easy_positives",
    "hard_positives",
    "easy_negatives",
    "hard_negatives",
]

GROUPS = {
    "easy":       ["easy_positives",  "easy_negatives"],
    "hard":       ["hard_positives",  "hard_negatives"],
    "same_topic": ["easy_positives",  "hard_negatives"],
    "diff_topic": ["hard_positives",  "easy_negatives"],
}

GROUP_TITLES = {
    "easy":       "Easy condition (EP + EN)",
    "hard":       "Hard condition (HP + HN)",
    "same_topic": "Same Topic  (easy pos + hard neg)",
    "diff_topic": "Different Topic  (hard pos + easy neg)",
}

TASK_LABELS = {
    "speaker": ("Same speaker",    "Different speaker"),
    "gender":  ("Same gender",     "Different gender"),
    "age":     ("Same age group",  "Different age group"),
    "accent":  ("Same accent",     "Different accent"),
}

MODEL_DISPLAY = {
    "sbert":     "SentenceBERT",
    "tfidf":     "TF-IDF",
    "cisr":      "CISR",
    "pcm":       "PCM",
    "luar":      "LUAR",
    "gpt":       "GPT-5-mini",
    "gpt-5-mini": "GPT-5-mini",
    "llama":     "Llama-3.3-70B",
    "qwen":      "Qwen3.5-9B",
}

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

N_BOOTSTRAP    = 100
BOOTSTRAP_FRAC = 0.80
FPR_TARGETS    = [0.01, 0.001]   # 1%  and  0.1%



def load_data(scored_dir: str, trial_dir: str) -> dict[str, pd.DataFrame]:
    """
    For each split, load the CSV (labels) and JSON (scores), join on trial_id.
    Returns dict  stem -> DataFrame  with columns [trial_id, label, <model>, ...]
    """
    data = {}
    for stem in TRIAL_FILES:
        csv_path  = os.path.join(trial_dir,  stem + ".csv")
        json_path = os.path.join(scored_dir, stem + ".json")

        if not os.path.exists(csv_path):
            print(f"  [skip] {stem}.csv not found")
            continue

        df = pd.read_csv(csv_path, dtype=str)
        df["label"] = df["label"].astype(int)

        if os.path.exists(json_path):
            with open(json_path, encoding="utf-8") as fh:
                scores = json.load(fh)
            score_df = pd.DataFrame.from_dict(scores, orient="index")
            score_df.index.name = "trial_id"
            score_df = score_df.reset_index()
            for col in score_df.columns:
                if col != "trial_id":
                    score_df[col] = pd.to_numeric(score_df[col], errors="coerce")
            df = df.merge(score_df, on="trial_id", how="left")
        else:
            print(f"  [warn] {stem}.json not found — no scores for this split")

        n_pos = int((df["label"] == 1).sum())
        n_neg = int((df["label"] == 0).sum())
        print(f"  {stem+'.csv':28s}  {len(df):>5,} rows  (+{n_pos:,} / -{n_neg:,})")
        data[stem] = df
    return data


def detect_models(data: dict[str, pd.DataFrame]) -> list[str]:
    """Return model keys present in any split, in display order."""
    found = set()
    for df in data.values():
        for col in df.columns:
            if col in MODEL_DISPLAY:
                if df[col].notna().sum() > 0:
                    found.add(col)
    ordered = [m for m in MODEL_DISPLAY if m in found]
    print(f"  Models found: {ordered}")
    return ordered



def compute_eer(fpr: np.ndarray, tpr: np.ndarray) -> float:
    fnr  = 1.0 - tpr
    diff = fnr - fpr
    idx  = np.nanargmin(np.abs(diff))
    if idx + 1 < len(diff) and diff[idx] * diff[idx + 1] < 0:
        slope = diff[idx + 1] - diff[idx]
        t     = -diff[idx] / slope if slope != 0 else 0.0
        eer   = fpr[idx] + t * (fpr[idx + 1] - fpr[idx])
    else:
        eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(np.clip(eer, 0.0, 1.0))


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float) -> float:
    """Interpolated TPR at a given FPR target."""
    return float(np.interp(target_fpr, fpr, tpr))


def compute_epsilon(fpr: np.ndarray, tpr: np.ndarray, n_trials: int) -> float:
    """
    VoxGuard epsilon:
        ε̂ = max{ max_t ln(TPR/FPR),  max_t ln((1-FPR)/(1-TPR)) }
    Each inner max is taken over all operating points where TPR >= 1/n_trials,
    then the outer max is taken between the two terms.
    """
    min_tpr = 1.0 / n_trials
    mask = (tpr >= min_tpr) & (fpr > 0) & (tpr < 1.0)
    if not mask.any():
        return float("nan")

    f = fpr[mask]
    t = tpr[mask]
    with np.errstate(divide="ignore", invalid="ignore"):
        eps1 = np.log(t / f)
        eps2 = np.log((1.0 - f) / (1.0 - t))

    max_eps1 = float(np.nanmax(eps1))
    max_eps2 = float(np.nanmax(eps2))
    result   = max(max_eps1, max_eps2)
    return result if np.isfinite(result) else float("nan")


def bootstrap_tpr_at_fpr(
    labels: np.ndarray,
    scores: np.ndarray,
    target_fpr: float,
    n_bootstrap: int = N_BOOTSTRAP,
    frac: float = BOOTSTRAP_FRAC,
    rng_seed: int = 42,
) -> dict:
    rng  = np.random.default_rng(rng_seed)
    n    = len(labels)
    k    = max(2, int(n * frac))
    vals = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, k, replace=False)
        lb, sc = labels[idx], scores[idx]
        if len(np.unique(lb)) < 2:
            continue
        f, t, _ = roc_curve(lb, sc)
        vals.append(tpr_at_fpr(f, t, target_fpr))
    if not vals:
        return {"mean": float("nan"), "std": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan")}
    arr = np.array(vals)
    return {
        "mean":    round(float(arr.mean()), 4),
        "std":     round(float(arr.std()),  4),
        "ci_low":  round(float(np.percentile(arr, 2.5)),  4),
        "ci_high": round(float(np.percentile(arr, 97.5)), 4),
    }


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict:
    """Full metric dict for one (model, view) pair."""
    mask   = ~np.isnan(scores)
    labels = labels[mask]
    scores = scores[mask]
    n      = len(labels)

    if n < 2 or len(np.unique(labels)) < 2:
        return None

    fpr, tpr, _ = roc_curve(labels, scores)
    auc         = float(roc_auc_score(labels, scores))
    eer         = compute_eer(fpr, tpr)
    eps         = compute_epsilon(fpr, tpr, n)

    result = {
        "n_trials": n,
        "n_pos":    int((labels == 1).sum()),
        "n_neg":    int((labels == 0).sum()),
        "auc":      round(auc, 4),
        "eer":      round(eer, 4),
        "epsilon":  round(eps, 4) if np.isfinite(eps) else None,
    }

    for target in FPR_TARGETS:
        key    = f"tpr_at_fpr_{target:.1%}".replace("%", "pct").replace(".", "_")
        result[key] = {
            "point_estimate": round(tpr_at_fpr(fpr, tpr, target), 4),
            "bootstrap":      bootstrap_tpr_at_fpr(labels, scores, target),
        }

    return result



def save_fig(fig, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def _style_ax(ax, title: str, pos_label: str, neg_label: str) -> None:
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random (AUC=0.50)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate",  fontsize=9)
    ax.set_title(f"{title}\n(+) {pos_label}  |  (-) {neg_label}", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)


def plot_roc(df: pd.DataFrame, models: list[str], title: str,
             pos_label: str, neg_label: str, out_path: str) -> dict:
    """
    Draw one ROC figure with all models. Returns metrics dict.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    metrics = {}

    for i, model in enumerate(models):
        if model not in df.columns:
            continue
        labels = df["label"].values.astype(int)
        scores = pd.to_numeric(df[model], errors="coerce").values
        mask   = ~np.isnan(scores)
        if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
            print(f"    [skip] {MODEL_DISPLAY[model]}: insufficient data")
            continue

        fpr, tpr, _ = roc_curve(labels[mask], scores[mask])
        auc = float(roc_auc_score(labels[mask], scores[mask]))
        eer = compute_eer(fpr, tpr)

        color = COLORS[i % len(COLORS)]
        ax.plot(fpr, tpr, color=color, lw=1.8,
                label=f"{MODEL_DISPLAY[model]}  AUC={auc:.3f}  EER={eer:.3f}")

        m = compute_metrics(labels, scores)
        if m:
            metrics[model] = m

    _style_ax(ax, title, pos_label, neg_label)
    fig.tight_layout()
    save_fig(fig, out_path)
    return metrics




def plot_roc_loglog(df: pd.DataFrame, models: list[str], title: str,
                    pos_label: str, neg_label: str, out_path: str) -> None:
    """
    ROC in log-log scale to reveal low-FPR behaviour (cf. VoxGuard Fig. 2).
    Axes: FPR in [1e-6, 1],  TPR in [1e-6, 1].
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # diagonal random-chance reference
    ref = np.logspace(-4, 0, 300)
    ax.plot(ref, ref, "k--", lw=0.8, label="Random")

    for i, model in enumerate(models):
        if model not in df.columns:
            continue
        labels = df["label"].values.astype(int)
        scores = pd.to_numeric(df[model], errors="coerce").values
        mask   = ~np.isnan(scores)
        if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
            continue

        fpr, tpr, _ = roc_curve(labels[mask], scores[mask])
        eer = compute_eer(fpr, tpr)

        # clip to positive values for log axes
        valid = (fpr > 0) & (tpr > 0)
        fpr_v, tpr_v = fpr[valid], tpr[valid]
        if len(fpr_v) < 2:
            continue

        color = COLORS[i % len(COLORS)]
        ax.plot(fpr_v, tpr_v, color=color, lw=1.8,
                label=f"{MODEL_DISPLAY[model]}  (EER={eer:.1%})")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-4, 1);  ax.set_ylim(1e-4, 1)
    ax.set_xlabel("FPR (log)", fontsize=9)
    ax.set_ylabel("TPR (log)", fontsize=9)
    ax.set_title(f"{title}\n(+) {pos_label}  |  (-) {neg_label}", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out_path)




def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SV trials from JSON score files.")
    parser.add_argument("--scored_dir", required=True,
                        help="Directory with easy/hard_positives/negatives .json files.")
    parser.add_argument("--trial_dir",  required=True,
                        help="Directory with matching .csv trial files.")
    parser.add_argument("--task", required=True, choices=list(TASK_LABELS),
                        help="Task type: speaker | gender | age | accent")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: eval/<task>/)")
    args = parser.parse_args()

    pos_label, neg_label = TASK_LABELS[args.task]
    output_dir = args.output_dir or os.path.join("eval", args.task)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  SV Evaluator  —  task={args.task.upper()}")
    print(f"{'='*60}")
    print(f"  scored_dir : {args.scored_dir}")
    print(f"  trial_dir  : {args.trial_dir}")
    print(f"  output_dir : {output_dir}")
    print(f"{'='*60}\n")

  
    print("[Loading data]")
    data   = load_data(args.scored_dir, args.trial_dir)
    if not data:
        print("No data found. Exiting."); return

    models = detect_models(data)
    if not models:
        print("No model scores found. Exiting."); return

    combined = pd.concat(data.values(), ignore_index=True)
    summary  = {}   # model -> view -> metrics


    print("\n[Combined ROC]")
    m = plot_roc(
        combined, models,
        title     = f"ROC — {args.task.upper()} (all trials)",
        pos_label = pos_label,
        neg_label = neg_label,
        out_path  = os.path.join(output_dir, "roc_combined.png"),
    )
    for model, metrics in m.items():
        summary.setdefault(model, {})["combined"] = metrics


    print("\n[Combined ROC — log scale]")
    plot_roc_loglog(
        combined, models,
        title     = f"ROC log-log — {args.task.upper()} (all trials)",
        pos_label = pos_label,
        neg_label = neg_label,
        out_path  = os.path.join(output_dir, "roc_combined_loglog.png"),
    )


    print("\n[Group ROCs]")
    for group_name, stems in GROUPS.items():
        dfs = [data[s] for s in stems if s in data]
        if not dfs:
            print(f"  [skip] {group_name}: no splits available"); continue
        df = pd.concat(dfs, ignore_index=True)
        if df["label"].nunique() < 2:
            print(f"  [skip] {group_name}: only one class is available")
            continue

        m = plot_roc(
            df, models,
            title     = f"ROC — {args.task.upper()}  |  {GROUP_TITLES[group_name]}",
            pos_label = pos_label,
            neg_label = neg_label,
            out_path  = os.path.join(output_dir, f"roc_{group_name}.png"),
        )
        for model, metrics in m.items():
            summary.setdefault(model, {})[group_name] = metrics

    named_summary = {
        MODEL_DISPLAY.get(model, model): views
        for model, views in summary.items()
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(named_summary, fh, indent=2)
    print(f"\n  Saved summary -> {summary_path}")


    print(f"\n{'='*80}")
    print(f"  Summary  [{args.task.upper()}]   AUC ↑   EER ↓   epsilon ↓")
    print(f"{'='*80}")
    views = ["combined"] + list(GROUPS.keys())
    header = f"  {'Model':15s}" + "".join(f"  {v:>12s}" for v in views)
    print(header + "  (metric)")
    for metric in ("auc", "eer"):
        print(f"  --- {metric.upper()} ---")
        for model, model_display in MODEL_DISPLAY.items():
            if model not in summary:
                continue
            row = f"  {model_display:15s}"
            for v in views:
                val = summary[model].get(v, {}).get(metric)
                row += f"  {val:>12.4f}" if val is not None else f"  {'—':>12s}"
            print(row)
    print(f"{'='*80}")

    print("\nDone.")


if __name__ == "__main__":
    main()