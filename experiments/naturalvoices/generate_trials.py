"""
Speaker Verification Trial Generator
======================================
Creates trials in three separate task subdirectories.

Task: speaker/
----------------------------------------------
  label = 1  same speaker
  label = 0  different speaker

  easy_positives  : same speaker, same podcast
  hard_positives  : same speaker, different podcast
  easy_negatives  : different speaker, different podcast
  hard_negatives  : different speaker, same podcast

Task: gender / age / accent
----------------------------------------------
  label = 1  same attribute (positive)
  label = 0  different attribute (negative)

  Pairs are always CROSS-SPEAKER.
  easy/hard is determined by whether probe and enroll
  come from the same podcast or different podcasts:

  easy_positives  : same podcast,      same attribute
  hard_positives  : different podcast, same attribute
  easy_negatives  : different podcast, different attribute
  hard_negatives  : same podcast,      different attribute

  Age bins   : <30 / 30-60 / 60+
  Accent bins: North America / Other

Every row carries the actual demographic values of both sides:
  probe_gender, probe_age_group, probe_accent_group,
  enroll_gender, enroll_age_group, enroll_accent_group

Usage
-----
    python trial_generator.py \\
        --input      data.json \\
        --output_dir trials \\
        --L          10
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Age binning
# ---------------------------------------------------------------------------

AGE_BINS   = [0, 30, 60, float("inf")]
AGE_LABELS = ["<30", "30-60", "60+"]


def age_to_group(age) -> str | None:
    if age is None:
        return None
    try:
        age = float(age)
    except (TypeError, ValueError):
        return None
    for low, high, label in zip(AGE_BINS, AGE_BINS[1:], AGE_LABELS):
        if low <= age < high:
            return label
    return None


# ---------------------------------------------------------------------------
# Accent binning
# ---------------------------------------------------------------------------

def accent_to_group(accent) -> str | None:
    if accent is None:
        return None
    return "North America" if str(accent).strip() == "North America" else "Other"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

META_KEYS = {"age", "gender", "accent", "accent_probs", "accent_num_utterances"}
REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_podcasts(spk_data: dict) -> list[str]:
    return [k for k in spk_data if k not in META_KEYS]


def collect_words_from_front(utterances: list[str], target_words: int, tolerance: int = 10):
    """Collect utterances from the front until the word count falls within
    [target_words - tolerance, target_words + tolerance].
    Returns ([], 0) if the window is overshot without hitting the range.
    """
    selected, total = [], 0
    for utt in utterances:
        selected.append(utt)
        total += len(utt.split())
        if total >= target_words - tolerance:
            if total <= target_words + tolerance:
                return selected, total # within range -> accept
            else:
                return [], 0 # overshot -> reject
    return [], 0 # ran out of utterances


def pack_utts(utts: list[str]) -> str:
    return " | ".join(utts)


def write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        print(f"    [skip] no trials -> {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Inject trial_id as first column
    split_name = Path(path).stem  # e.g. "easy_positives"
    for i, row in enumerate(rows):
        row["trial_id"] = f"{split_name}_{i:06d}"

    all_keys   = list(rows[0].keys())
    fieldnames = ["trial_id"] + [k for k in all_keys if k != "trial_id"]

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {k: pack_utts(v) if isinstance(v, list) else v
                    for k, v in row.items()}
            writer.writerow(flat)
    n1 = sum(r["label"] == 1 for r in rows)
    n0 = sum(r["label"] == 0 for r in rows)
    print(f"    {len(rows):>6,} rows  (+{n1:,} / -{n0:,})  -> {path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(json_path: str, max_speakers: int, seed: int) -> dict:
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)
    speakers = list(data.keys())
    random.seed(seed)
    if len(speakers) > max_speakers:
        speakers = random.sample(speakers, max_speakers)
        data = {s: data[s] for s in speakers}
    print(f"  Loaded {len(data):,} speakers")
    return data


def get_demo(spk_data: dict) -> tuple:
    gender = spk_data.get("gender")
    age_group = age_to_group(spk_data.get("age"))
    accent_group = accent_to_group(spk_data.get("accent"))
    return (str(gender).lower().strip() if gender else None, age_group, accent_group)


def make_row(probe_spk, probe_pod, probe_utts,
             enroll_spk, enroll_pod, enroll_utts,
             data: dict) -> dict:
    """Base row with all demographic columns. label=None, filled per task."""
    pg, pa, pac = get_demo(data[probe_spk])
    eg, ea, eac = get_demo(data[enroll_spk])
    return {
        "label": None,
        "probe_speaker": probe_spk,
        "probe_podcast": probe_pod,
        "probe_utterances": probe_utts,
        "probe_gender": pg,
        "probe_age_group": pa,
        "probe_accent_group": pac,
        "enroll_speaker": enroll_spk,
        "enroll_podcast": enroll_pod,
        "enroll_utterances": enroll_utts,
        "enroll_gender": eg,
        "enroll_age_group": ea,
        "enroll_accent_group": eac,
    }


# ---------------------------------------------------------------------------
# Base trial builders  (speaker-identity based, labels set later)
# ---------------------------------------------------------------------------

def build_same_speaker_same_podcast(data, L, probe_words, seed, tolerance) -> list:
    random.seed(seed)
    rows = []
    for spk, spk_data in data.items():
        for pod in get_podcasts(spk_data):
            utts = spk_data[pod]
            if len(utts) < 2:
                continue
            mid = len(utts) // 2
            probe_utts, _ = collect_words_from_front(utts[:mid], probe_words, tolerance)
            if not probe_utts:
                continue
            second = utts[mid:]
            enroll_utts = second[-L:] if len(second) >= L else second
            if not enroll_utts:
                continue
            rows.append(make_row(spk, pod, probe_utts,
                                 spk, pod, enroll_utts, data))
    return rows


def build_same_speaker_diff_podcast(data, L, probe_words, seed, tolerance) -> list:
    random.seed(seed)
    rows = []
    for spk, spk_data in data.items():
        podcasts = get_podcasts(spk_data)
        if len(podcasts) < 2:
            continue
        for probe_pod in podcasts:
            probe_utts, _ = collect_words_from_front(
                spk_data[probe_pod], probe_words, tolerance)
            if not probe_utts:
                continue
            enroll_pod  = random.choice([p for p in podcasts if p != probe_pod])
            enroll_utts = spk_data[enroll_pod][:L]
            if not enroll_utts:
                continue
            rows.append(make_row(spk, probe_pod, probe_utts,
                                 spk, enroll_pod, enroll_utts, data))
    return rows


def build_diff_speaker_diff_podcast(data, L, probe_words, seed, tolerance) -> list:
    random.seed(seed)
    combos = [
        (spk, pod, spk_data[pod])
        for spk, spk_data in data.items()
        for pod in get_podcasts(spk_data)
    ]
    pod_index = defaultdict(list)
    for spk, pod, utts in combos:
        pod_index[pod].append((spk, utts))
    all_pods = list(pod_index.keys())

    rows = []
    for probe_spk, probe_pod, probe_all in combos:
        probe_utts, _ = collect_words_from_front(probe_all, probe_words, tolerance)
        if not probe_utts:
            continue
        candidates = [p for p in all_pods if p != probe_pod]
        if not candidates:
            continue
        random.shuffle(candidates)
        chosen = None
        for cand_pod in candidates:
            eligible = [(s, u) for s, u in pod_index[cand_pod]
                        if s != probe_spk]
            if eligible:
                enroll_spk, enroll_all = random.choice(eligible)
                chosen = (enroll_spk, cand_pod, enroll_all[:L])
                break
        if chosen is None:
            continue
        enroll_spk, enroll_pod, enroll_utts = chosen
        rows.append(make_row(probe_spk, probe_pod, probe_utts,
                             enroll_spk, enroll_pod, enroll_utts, data))
    return rows


def build_diff_speaker_same_podcast(data, L, probe_words, seed, tolerance) -> list:
    random.seed(seed)
    pod_index = defaultdict(list)
    for spk, spk_data in data.items():
        for pod in get_podcasts(spk_data):
            pod_index[pod].append((spk, spk_data[pod]))

    rows = []
    for spk, spk_data in data.items():
        for pod in get_podcasts(spk_data):
            probe_utts, _ = collect_words_from_front(spk_data[pod], probe_words, tolerance)
            if not probe_utts:
                continue
            others = [(s, u) for s, u in pod_index[pod] if s != spk]
            if not others:
                continue
            enroll_spk, enroll_all = random.choice(others)
            rows.append(make_row(spk, pod, probe_utts,
                                 enroll_spk, pod, enroll_all[:L], data))
    return rows


# ---------------------------------------------------------------------------
# Task writers
# ---------------------------------------------------------------------------

def write_speaker_task(base, output_dir: str) -> None:
    """Speaker task: label = same speaker identity."""
    task_dir = os.path.join(output_dir, "speaker")
    print(f"\n[Task: speaker]  ->  {task_dir}/")
    mapping = {
        "easy_positives.csv": ("same_same", 1),
        "hard_positives.csv": ("same_diff", 1),
        "easy_negatives.csv": ("diff_diff", 0),
        "hard_negatives.csv": ("diff_same", 0),
    }
    for fname, (key, lbl) in mapping.items():
        rows = [{**r, "label": lbl} for r in base[key]]
        write_csv(rows, os.path.join(task_dir, fname))


def write_demo_task(base, output_dir: str, task_name: str, attr_key: str) -> None:
    """
    Demographic attribute task (gender / age / accent).
      - All pairs are CROSS-SPEAKER
      - easy = same podcast,      hard = different podcast
      - positives = same attribute, negatives = different attribute

    easy_positives  : same podcast      + same attribute
    hard_positives  : different podcast + same attribute
    easy_negatives  : different podcast + different attribute
    hard_negatives  : same podcast      + different attribute
    """
    task_dir = os.path.join(output_dir, task_name)
    print(f"\n[Task: {task_name}]  ->  {task_dir}/")

    all_cross_speaker = [
        r for key in ("diff_diff", "diff_same")
        for r in base[key]
    ]
    print(f"  Cross-speaker pool: {len(all_cross_speaker):,} rows")

    easy_pool = [r for r in all_cross_speaker
                 if r["probe_podcast"] == r["enroll_podcast"]]
    hard_pool = [r for r in all_cross_speaker
                 if r["probe_podcast"] != r["enroll_podcast"]]

    def attr_match(r) -> bool:
        pv = r.get(f"probe_{attr_key}")
        ev = r.get(f"enroll_{attr_key}")
        return pv is not None and ev is not None and pv == ev

    for fname, pool, keep_match in [
        ("easy_positives.csv", easy_pool, True),   # same podcast,      same attr
        ("hard_positives.csv", hard_pool, True),   # different podcast, same attr
        ("easy_negatives.csv", hard_pool, False),  # different podcast, diff attr
        ("hard_negatives.csv", easy_pool, False),  # same podcast,      diff attr
    ]:
        rows = [
            {**r, "label": 1 if keep_match else 0}
            for r in pool
            if attr_match(r) == keep_match
        ]
        write_csv(rows, os.path.join(task_dir, fname))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SV trials for speaker / gender / age / accent tasks.")
    parser.add_argument("--input", default=REPO_ROOT / "data_processing" / "naturalvoices" / "data" / "speaker_info.json")
    parser.add_argument("--output_dir", default="trials")
    parser.add_argument("--L", type=int, default=10)
    parser.add_argument("--max_speakers", type=int, default=2000)
    parser.add_argument("--probe_words", type=int, default=200)
    parser.add_argument("--tolerance", type=int, default=10,
                        help="Probe word count tolerance: target ± tolerance (default: 10).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  SV Trial Generator - Speaker / Gender / Age / Accent")
    print(f"{'='*60}")
    print(f"  Input : {args.input}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  L (enroll) : {args.L}")
    print(f"  max_speakers : {args.max_speakers}")
    print(f"  probe_words : {args.probe_words}")
    print(f"  seed : {args.seed}")
    print(f"  tolerance : {args.tolerance}")
    print(f"{'='*60}\n")

    data = load_data(args.input, args.max_speakers, args.seed)

    print("\n[Building base trial sets]")
    builders = {
        "same_same": (build_same_speaker_same_podcast, "same speaker / same podcast"),
        "same_diff": (build_same_speaker_diff_podcast, "same speaker / diff podcast"),
        "diff_diff": (build_diff_speaker_diff_podcast, "diff speaker / diff podcast"),
        "diff_same": (build_diff_speaker_same_podcast, "diff speaker / same podcast"),
    }
    base = {}
    for key, (fn, desc) in builders.items():
        base[key] = fn(data, L=args.L, probe_words=args.probe_words, seed=args.seed, tolerance=args.tolerance)
        print(f"  {desc:35s}  {len(base[key]):,} rows")

    write_speaker_task(base, args.output_dir)
    write_demo_task(base, args.output_dir, "gender", "gender")
    write_demo_task(base, args.output_dir, "age", "age_group")
    write_demo_task(base, args.output_dir, "accent", "accent_group")

    print(f"\n{'='*60}")
    print("  Output layout")
    print(f"{'='*60}")
    for task, desc in [
        ("speaker", "label=1 same speaker / label=0 diff speaker"),
        ("gender", "label=1 same gender  / label=0 diff gender  (cross-speaker only)"),
        ("age", "label=1 same age grp / label=0 diff age grp  (cross-speaker only)"),
        ("accent", "label=1 same accent  / label=0 diff accent  (cross-speaker only)"),
    ]:
        print(f"  {args.output_dir}/{task}/  <- {desc}")
    print()


if __name__ == "__main__":
    main()