"""
Scorer for speaker-verification trials.

Models
------------------
    sbert      – SentenceBERT (all-mpnet-base-v2)
    tfidf      – TF-IDF cosine similarity
    cisr       – CISR / Style-Embedding (AnnaWegmann/Style-Embedding)
    pcm        – PCM style (Blablablab/multilingual-style-representation)
    luar       – LUAR-CRUD 
    gpt        – GPT-5-mini 
    llama      – Llama-3.3-70B-Instruct-Turbo  
    qwen       – Qwen3.5-9B  

Output
------
Up to four JSON files (one per available trial split), accumulating across runs:

    {
        "easy_positives_000000": {
            "sbert": 0.812,
            "luar":  0.754,
            "gpt":   78.0,
            ...
        },
        ...
    }

Usage
-----
    python score_trials.py \\
        --trial_dir  trials/speaker/ \\
        --output_dir scored/ \\
        --models sbert tfidf luar \\
        --task   speaker \\
        --device cuda

    python score_trials.py \\
        --trial_dir  trials/gender/ \\
        --output_dir scored/ \\
        --models gpt llama qwen \\
        --task   gender \\
        --llm_mode binary
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

TRIAL_FILES = [
    "easy_positives.csv",
    "hard_positives.csv",
    "easy_negatives.csv",
    "hard_negatives.csv",
]

EMBEDDING_MODELS = ["sbert", "tfidf", "cisr", "pcm", "luar"]
LLM_MODELS = ["gpt", "llama", "qwen"]
ALL_MODELS = EMBEDDING_MODELS + LLM_MODELS

TOGETHER_MODEL_MAP = {
    "llama": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "qwen":  "Qwen/Qwen3.5-9B",
}
TOGETHER_BASE_URL = "https://api.together.ai/v1"
GPT_MODEL = "gpt-5-mini"

TASKS = ["speaker", "age", "gender", "accent"]


# -------------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------------

def parse_utts(cell: Any) -> list[str]:
    return [u.strip() for u in str(cell).split(" | ") if u.strip()]


def format_utterances(utterances: list[str]) -> str:
    return "\n".join(f"{i + 1}. {utt}" for i, utt in enumerate(utterances))


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    return float(sk_cosine(a.reshape(1, -1), b.reshape(1, -1))[0, 0])


def retry_call(fn, *, retries: int = 4, base_sleep: float = 1.5):
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as err:  # noqa: BLE001
            last_err = err
            status_code = getattr(err, "status_code", None)
            if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                break
            if attempt < retries - 1:
                wait = base_sleep * (2 ** attempt)
                print(f"    [retry {attempt + 1}/{retries}] {err} — waiting {wait:.1f}s")
                time.sleep(wait)
    raise last_err  # type: ignore[misc]


# -------------------------------------------------------------------------
# JSON score store
# -------------------------------------------------------------------------

def load_store(json_path: str) -> dict[str, dict]:
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _merge_scores(base: dict[str, dict], update: dict[str, dict]) -> None:
    for trial_id, scores in update.items():
        if not isinstance(scores, dict):
            continue
        target = base.setdefault(trial_id, {})
        for model, value in scores.items():
            if value is None and target.get(model) is not None:
                continue
            target[model] = value


def save_store(store: dict[str, dict], json_path: str) -> None:
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    lock_path = json_path + ".lock"
    with open(lock_path, "a+", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        latest = load_store(json_path)
        _merge_scores(latest, store)

        tmp_path = f"{json_path}.{os.getpid()}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(latest, fh, indent=2)
        os.replace(tmp_path, json_path)

        store.clear()
        store.update(latest)


# -------------------------------------------------------------------------
# Embedding model loaders
# -------------------------------------------------------------------------

def load_sbert(device: str):
    from sentence_transformers import SentenceTransformer
    print("  Loading SBERT (all-mpnet-base-v2) ...")
    m = SentenceTransformer("all-mpnet-base-v2", device=device)
    print(f"  SBERT ready on {device}")
    return m


def load_cisr(device: str):
    from sentence_transformers import SentenceTransformer
    print("  Loading CISR (AnnaWegmann/Style-Embedding) ...")
    m = SentenceTransformer("AnnaWegmann/Style-Embedding", device=device)
    print(f"  CISR ready on {device}")
    return m


def load_pcm(device: str):
    from sentence_transformers import SentenceTransformer
    print("  Loading PCM (Blablablab/multilingual-style-representation) ...")
    m = SentenceTransformer("Blablablab/multilingual-style-representation", device=device)
    print(f"  PCM ready on {device}")
    return m



def load_luar(device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer
    print("  Loading LUAR (rrivera1849/LUAR-CRUD) ...")
    tokenizer = AutoTokenizer.from_pretrained(
        "rrivera1849/LUAR-CRUD", trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        "rrivera1849/LUAR-CRUD", trust_remote_code=True).to(device)
    model.eval()
    print(f"  LUAR ready on {device}")
    return model, tokenizer


def fit_tfidf(trial_dir: str) -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer
    print("  Fitting TF-IDF ...")
    utts: set[str] = set()
    for fname in TRIAL_FILES:
        fpath = os.path.join(trial_dir, fname)
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath, dtype=str)
        for col in ("probe_utterances", "enroll_utterances"):
            if col in df.columns:
                for cell in df[col].dropna():
                    utts.update(parse_utts(cell))
    vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        min_df=2, max_features=50_000, sublinear_tf=True,
    )
    vec.fit(list(utts))
    print(f"  TF-IDF fitted on {len(utts):,} utterances")
    return vec


# -------------------------------------------------------------------------
# Embedding scorers
# -------------------------------------------------------------------------

def _sbert_embed(utterances: list[str], model, batch_size: int) -> np.ndarray:
    embs = model.encode(
        utterances, batch_size=batch_size,
        show_progress_bar=False, convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embs.mean(axis=0)


def _luar_embed(utterances: list[str], model, tokenizer,
                episode_length: int, max_length: int, device: str) -> np.ndarray:
    import torch
    utts = utterances[:episode_length]
    if len(utts) < episode_length:
        utts = utts + [utts[-1]] * (episode_length - len(utts))
    tok = tokenizer(utts, max_length=max_length, padding="max_length",
                    truncation=True, return_tensors="pt")
    input_ids = tok["input_ids"].reshape(1, episode_length, -1).to(device)
    attention_mask = tok["attention_mask"].reshape(1, episode_length, -1).to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.squeeze(0).cpu().numpy()


def _tfidf_embed(utterances: list[str], vec) -> np.ndarray:
    mat = vec.transform(utterances)
    return np.asarray(mat.mean(axis=0)).squeeze()


def score_embedding_model(
    key: str,
    df: pd.DataFrame,
    store: dict[str, dict],
    models: dict,
    args,
    json_path: str,
) -> None:
    """Generic embedding scorer; dispatches by key."""
    pending = [
        (idx, row) for idx, row in df.iterrows()
        if store.get(row["trial_id"], {}).get(key) is None
    ]
    print(f"    {key}: {len(pending):,} rows to score")

    for position, (_, row) in enumerate(tqdm(pending, desc=f"    {key}"), start=1):
        tid = row["trial_id"]
        probe_utts = parse_utts(row["probe_utterances"])
        enroll_utts = parse_utts(row["enroll_utterances"])

        if key in ("sbert", "cisr", "pcm"):
            model = models[key]
            ep = _sbert_embed(probe_utts,  model, args.batch_size)
            ee = _sbert_embed(enroll_utts, model, args.batch_size)
            score = cosine_np(ep, ee)

        elif key == "luar":
            model, tokenizer = models["luar"]
            ep = _luar_embed(probe_utts,  model, tokenizer,
                             args.episode_length, args.max_length, args.device)
            ee = _luar_embed(enroll_utts, model, tokenizer,
                             args.episode_length, args.max_length, args.device)
            import torch
            import torch.nn.functional as F
            ta, tb = torch.tensor(ep), torch.tensor(ee)
            score = float(F.cosine_similarity(ta.unsqueeze(0), tb.unsqueeze(0)).item())

        elif key == "tfidf":
            vec = models["tfidf"]
            ep = _tfidf_embed(probe_utts,  vec)
            ee = _tfidf_embed(enroll_utts, vec)
            score = cosine_np(ep, ee)

        else:
            raise ValueError(f"Unknown embedding model key: {key}")

        store.setdefault(tid, {})[key] = score
        if position % 100 == 0:
            save_store(store, json_path)
    if pending:
        save_store(store, json_path)


# -------------------------------------------------------------------------
# LLM prompts  (task-aware)
# -------------------------------------------------------------------------

_TASK_BINARY_PROMPTS = {
    "speaker": (
        "Decide whether the probe and enrollment utterances were written "
        "or spoken by the SAME SPEAKER.\n"
        "Focus on style, wording, phrasing habits, and discourse patterns.\n"
        "Output exactly one token:\n1 = same speaker\n0 = different speaker"
    ),
    "gender": (
        "Decide whether the probe and enrollment utterances were produced "
        "by speakers of the SAME GENDER.\n"
        "Focus on stylistic and linguistic cues.\n"
        "Output exactly one token:\n1 = same gender\n0 = different gender"
    ),
    "age": (
        "Decide whether the probe and enrollment utterances were produced "
        "by speakers of the SAME AGE GROUP (groups: <30 / 30-60 / 60+).\n"
        "Focus on stylistic and linguistic cues.\n"
        "Output exactly one token:\n1 = same age group\n0 = different age group"
    ),
    "accent": (
        "Decide whether the probe and enrollment utterances were produced "
        "by speakers with the SAME ACCENT GROUP (North America vs Other).\n"
        "Focus on vocabulary, phrasing, and linguistic markers.\n"
        "Output exactly one token:\n1 = same accent group\n0 = different accent group"
    ),
}

_TASK_SCORE_PROMPTS = {
    "speaker": (
        "Estimate how likely it is that the probe and enrollment utterances "
        "came from the SAME SPEAKER.\n"
        "Focus on stylistic and linguistic cues, not topic similarity.\n"
        "Output one number from 0 to 100:\n"
        "0 = definitely different speaker\n100 = definitely same speaker"
    ),
    "gender": (
        "Estimate how likely it is that the probe and enrollment utterances "
        "came from speakers of the SAME GENDER.\n"
        "Focus on stylistic and linguistic cues.\n"
        "Output one number from 0 to 100:\n"
        "0 = definitely different gender\n100 = definitely same gender"
    ),
    "age": (
        "Estimate how likely it is that the probe and enrollment utterances "
        "came from speakers of the SAME AGE GROUP (groups: <30 / 30-60 / 60+).\n"
        "Focus on stylistic and linguistic cues.\n"
        "Output one number from 0 to 100:\n"
        "0 = definitely different age group\n100 = definitely same age group"
    ),
    "accent": (
        "Estimate how likely it is that the probe and enrollment utterances "
        "came from speakers with the SAME ACCENT GROUP (North America vs Other).\n"
        "Focus on vocabulary, phrasing, and linguistic markers.\n"
        "Output one number from 0 to 100:\n"
        "0 = definitely different accent group\n100 = definitely same accent group"
    ),
}


def _build_prompt(task: str, mode: str,
                  probe: list[str], enroll: list[str]) -> str:
    instruction = (
        _TASK_BINARY_PROMPTS[task] if mode == "binary"
        else _TASK_SCORE_PROMPTS[task]
    )
    return (
        f"You are a speaker verification judge.\n\n"
        f"{instruction}\n\n"
        f"Do not explain your reasoning.\n\n"
        f"Probe utterances:\n{format_utterances(probe)}\n\n"
        f"Enrollment utterances:\n{format_utterances(enroll)}"
    )


def _parse_binary(text: str) -> int | None:
    norm = text.strip().lower()
    if norm.startswith("1") or norm.startswith("yes"):
        return 1
    if norm.startswith("0") or norm.startswith("no"):
        return 0
    m = re.search(r"\b([01])\b", norm)
    return int(m.group(1)) if m else None


def _parse_score(text: str) -> float | None:
    first_line = text.split("\n")[0]
    for haystack in (first_line, text):
        m = re.search(r"\b(\d{1,3}(?:\.\d+)?)\b", haystack.replace(",", " "))
        if m:
            return float(max(0.0, min(100.0, float(m.group(1)))))
    return None


# -------------------------------------------------------------------------
# LLM scorers
# -------------------------------------------------------------------------

def score_gpt(
    df: pd.DataFrame,
    store: dict[str, dict],
    client,
    task: str,
    mode: str,
    pause_s: float,
    json_path: str,
) -> None:
    pending = [
        (idx, row) for idx, row in df.iterrows()
        if store.get(row["trial_id"], {}).get("gpt") is None
    ]
    print(f"    gpt: {len(pending):,} rows to score")
    wrote = 0
    errors = 0
    parse_errors = 0

    for _, row in tqdm(pending, desc="    gpt"):
        tid = row["trial_id"]
        probe_utts = parse_utts(row["probe_utterances"])
        enroll_utts = parse_utts(row["enroll_utterances"])
        prompt = _build_prompt(task, mode, probe_utts, enroll_utts)

        try:
            def _call():
                return client.responses.create(
                    model=GPT_MODEL,
                    input=prompt,
                    reasoning={"effort": "low"},
                    text={"verbosity": "low"},
                )
            response = retry_call(_call)

            # extract text from response
            text = getattr(response, "output_text", None)
            if not isinstance(text, str) or not text.strip():
                parts = []
                for item in getattr(response, "output", []) or []:
                    for content in getattr(item, "content", []) or []:
                        if getattr(content, "type", None) == "output_text":
                            parts.append(getattr(content, "text", ""))
                text = "".join(parts).strip()

            parsed = _parse_binary(text) if mode == "binary" else _parse_score(text)
            if parsed is None:
                parse_errors += 1
                print(f"\n  [parse-error] {tid}: {text[:200]!r} — skipping.")
                continue
            store.setdefault(tid, {})["gpt"] = parsed
            save_store(store, json_path)
            wrote += 1

        except Exception as e:  # noqa: BLE001
            errors += 1
            print(f"\n  [error] {tid}: {e} — skipping.")

        if pause_s > 0:
            time.sleep(pause_s)

    if pending:
        print(
            f"    gpt summary: wrote={wrote:,}, "
            f"errors={errors:,}, parse_errors={parse_errors:,}"
        )
    if pending and wrote == 0:
        raise RuntimeError(
            f"gpt produced no new scores for {json_path}; "
            "check the API errors above."
        )


def score_together_model(
    key: str,
    df: pd.DataFrame,
    store: dict[str, dict],
    client,
    task: str,
    mode: str,
    pause_s: float,
    json_path: str,
) -> None:
    model_name  = TOGETHER_MODEL_MAP[key]
    max_tokens  = 16 if mode == "binary" else 256
    pending = [
        (idx, row) for idx, row in df.iterrows()
        if store.get(row["trial_id"], {}).get(key) is None
    ]
    print(f"    {key}: {len(pending):,} rows to score")
    wrote = 0
    errors = 0
    parse_errors = 0

    for _, row in tqdm(pending, desc=f"    {key}"):
        tid = row["trial_id"]
        probe_utts = parse_utts(row["probe_utterances"])
        enroll_utts = parse_utts(row["enroll_utterances"])
        prompt = _build_prompt(task, mode, probe_utts, enroll_utts)

        try:
            def _call():
                kwargs = dict(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
                if key == "qwen":
                    kwargs.update(
                        extra_body={"reasoning": {"enabled": False}},
                        temperature=1.0,
                        top_p=0.95,
                        top_k=20,
                        min_p=0.0,
                        presence_penalty=1.5,
                        repetition_penalty=1.0,
                    )
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content

            text = retry_call(_call).strip()
            parsed = _parse_binary(text) if mode == "binary" else _parse_score(text)
            if parsed is None:
                parse_errors += 1
                print(f"\n  [parse-error:{key}] {tid}: {text[:200]!r} — skipping.")
                continue
            store.setdefault(tid, {})[key] = parsed
            save_store(store, json_path)
            wrote += 1

        except Exception as e:  # noqa: BLE001
            errors += 1
            print(f"\n  [error:{key}] {tid}: {e} — skipping.")

        if pause_s > 0:
            time.sleep(pause_s)

    if pending:
        print(
            f"    {key} summary: wrote={wrote:,}, "
            f"errors={errors:,}, parse_errors={parse_errors:,}"
        )
    if pending and wrote == 0:
        raise RuntimeError(
            f"{key} produced no new scores for {json_path}; "
            "check the API errors above."
        )


def score_together_models_sequentially(
    keys: list[str],
    df: pd.DataFrame,
    api_key: str,
    task: str,
    mode: str,
    pause_s: float,
    json_path: str,
) -> None:
    """Score Together-hosted models one at a time to respect rate limits."""
    from together import Together

    for key in keys:
        print(f"    Running Together model: {key}")
        client = Together(api_key=api_key, base_url=TOGETHER_BASE_URL)
        model_store = load_store(json_path)
        score_together_model(
            key, df, model_store, client,
            task, mode, pause_s, json_path,
        )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score SV trials with selected models; results accumulate to JSON.")

    parser.add_argument("--trial_dir",  default="trials",
                        help="Directory containing trial CSV files.")
    parser.add_argument("--output_dir", default="scored",
                        help="Directory to write / update score JSON files.")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS + ["all"],
                        required=True,
                        help="Models to run. Choose from: " + ", ".join(ALL_MODELS) + ", all")
    parser.add_argument("--task", choices=TASKS, required=True,
                        help="Trial task type — drives LLM prompts.")
    parser.add_argument("--llm_mode", choices=("score", "binary"), default="score",
                        help="LLM output mode: 'score' (0-100) or 'binary' (0/1). Default: score.")

    # Embedding
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for SentenceTransformer models.")
    parser.add_argument("--episode_length", type=int, default=16,
                        help="Episode length for LUAR.")
    parser.add_argument("--max_length", type=int, default=32,
                        help="Max token length per utterance for LUAR.")

    # API keys

    # LLM pacing
    parser.add_argument("--pause_s", type=float, default=0.0,
                        help="Seconds to wait between LLM API calls.")

    args = parser.parse_args()

    selected: list[str] = ALL_MODELS if "all" in args.models else list(dict.fromkeys(args.models))

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("  SV Trial Scorer (unified)")
    print(f"{'='*60}")
    print(f"  trial_dir : {args.trial_dir}")
    print(f"  output_dir : {args.output_dir}")
    print(f"  models : {selected}")
    print(f"  task : {args.task}")
    print(f"  llm_mode : {args.llm_mode}")
    print(f"{'='*60}\n")

    # ── Load models ───────────────────────────────────────────────────
    loaded: dict = {}

    if "sbert"    in selected: loaded["sbert"] = load_sbert(args.device)
    if "cisr"     in selected: loaded["cisr"] = load_cisr(args.device)
    if "pcm"      in selected: loaded["pcm"] = load_pcm(args.device)
    if "luar"     in selected: loaded["luar"] = load_luar(args.device)
    if "tfidf"    in selected: loaded["tfidf"] = fit_tfidf(args.trial_dir)

    gpt_client = None
    together_api_key = None

    if "gpt" in selected:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        gpt_client = OpenAI(api_key=api_key)

    if any(k in selected for k in ("llama", "qwen")):
        together_api_key = (
            os.environ.get("TOGETHER_API_KEY", "")
        )
        if not together_api_key:
            raise EnvironmentError("TOGETHER_API_KEY is not set.")

    # ── Score each trial file ─────────────────────────────────────────
    trial_parts = Path(args.trial_dir).parts
    probe_lsize = trial_parts[-2] if len(trial_parts) >= 2 else "unknown"
    scored_subdir = os.path.join(args.output_dir, f"{args.task}_{probe_lsize}")
    os.makedirs(scored_subdir, exist_ok=True)

    for fname in TRIAL_FILES:
        csv_path  = os.path.join(args.trial_dir, fname)
        json_path = os.path.join(scored_subdir,  fname.replace(".csv", ".json"))

        if not os.path.exists(csv_path):
            print(f"  [skip] {fname} not found in {args.trial_dir}")
            continue

        print(f"\n[{fname}]")
        df    = pd.read_csv(csv_path, dtype=str)
        store = load_store(json_path)

        if "trial_id" not in df.columns:
            raise ValueError(
                f"{csv_path} is missing a 'trial_id' column. "
                "Re-generate your trials with the updated trial_generator.py.")

        # Embedding models
        for key in EMBEDDING_MODELS:
            if key in selected:
                score_embedding_model(key, df, store, loaded, args, json_path)

        # LLM models
        if "gpt" in selected:
            score_gpt(df, store, gpt_client, args.task, args.llm_mode, args.pause_s, json_path)

        together_keys = [
            key for key in ("llama", "qwen") if key in selected
        ]
        if together_keys:
            score_together_models_sequentially(
                together_keys, df, together_api_key,
                args.task, args.llm_mode, args.pause_s, json_path,
            )

        # ── Summary ───────────────────────────────────────────────────
        store = load_store(json_path)
        print(f"  {len(store):,} trials  →  {json_path}")
        for k in selected:
            vals = [v[k] for v in store.values() if k in v and v[k] is not None]
            if vals:
                arr = np.array(vals, dtype=float)
                print(f"    {k:12s}  mean={arr.mean():.4f}  std={arr.std():.4f}  n={len(arr):,}")

    print("\nDone.")


if __name__ == "__main__":
    main()
