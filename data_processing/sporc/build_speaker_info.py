"""
Build per-speaker SPoRC info from:
  - sporc-single-speaker.jsonl  (for hostPredictedNames per episode)
  - filtered_output/ep_XXXXXX.json  (for transcripts per segment)
  - utterance_labels/age_sex.shard_*_of_*.json  (merged across shards)
  - utterance_labels/dialect.shard_*_of_*.json  (merged across shards)

Output: sporc_speaker_info.json

Structure:
{
  "Speaker Name": {
    "ep_XXXXXX": ["canary_text of seg_0000", "... of seg_0002", ...],
    "ep_YYYYYY": [...],
    "age":                    <float years>  or  None,
    "gender":                 "female" | "male" | "Unknown",
    "accent":                 "<dialect label>"  or  "Unknown",
    "accent_probs":           {"Eastern Asia": 0.01, "English": 0.42, ...}  or  {},
    "accent_num_utterances":  <int>,
  },
  ...
}

Grouping:
  Only episodes with exactly one hostPredictedNames entry are included.
  Multi-host episodes are skipped. Remaining episodes are grouped by that
  single host name, so all episodes under the same name are merged into one
  speaker — no automatic name-collision splitting. If you need to disambiguate
  two real people who share a name, that's a manual post-processing step.

Aggregation (per speaker, across all their surviving utterances):
  - age:    arithmetic mean of per-utterance age predictions
  - gender: argmax of mean [female_prob, male_prob] vector
  - accent: argmax of mean 16-dim dialect probability vector

Episode indexing:
  ep_id = f"ep_{line_index:06d}" where line_index is the 0-based line in the
  jsonl. This matches how sporc_data_filter.py names filter outputs (e.g.
  segment with episode_index=19 -> ep_000019.json -> line 20 of the jsonl).
"""

import argparse
import glob
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

DIALECT_LABEL_LIST = [
    "Eastern Asia", "English", "Germanic", "Irish",
    "North America", "Northern Irish", "Oceania",
    "Other", "Romance", "Scottish", "Semitic", "Slavic",
    "South African", "South-eastern Asia", "Southern Asia", "Welsh",
]

# The age/sex checkpoint uses probability index 0 = Female, 1 = Male.
# Downcased for consistency with the NaturalVoices speaker_info convention.
SEX_LABEL_LIST = ["female", "male"]


# ----------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------
def merge_shards(shard_glob: str) -> dict:
    """Merge all per-shard JSON files matching shard_glob into one dict."""
    merged = {}
    for path in sorted(glob.glob(shard_glob)):
        with open(path) as f:
            merged.update(json.load(f))
    return merged


def load_episode_to_hosts(jsonl_path: str) -> dict:
    """
    Build ep_id -> list-of-host-names from the SPoRC jsonl.

    Uses 0-based line index to match sporc_data_filter.py's episode_index.
    """
    mapping = {}
    with open(jsonl_path) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                logging.warning(f"jsonl line {idx}: JSON error {e}, skipping")
                continue
            ep_id = f"ep_{idx:06d}"
            names = rec.get("hostPredictedNames", [])
            # hostPredictedNames is usually a list; sometimes a sentinel string
            # like "NO_HOST_PREDICTED". Normalize to list of strings.
            if isinstance(names, str):
                names = [names] if names and names != "NO_HOST_PREDICTED" else []
            mapping[ep_id] = names
    return mapping


# ----------------------------------------------------------------------
# Utterance-level probability extraction (robust to one-layer nesting)
# ----------------------------------------------------------------------
def _unwrap(prob):
    """softmax(..., dim=1).tolist() gives [[p0, p1, ...]]. Unwrap one layer if needed."""
    if isinstance(prob, list) and len(prob) > 0 and isinstance(prob[0], list):
        return prob[0]
    return prob


def extract_dialect_probs(ep_dialect_entry, seg_id):
    """Return 1-D np.array(16,) of dialect probabilities for seg_id, or None."""
    if seg_id not in ep_dialect_entry:
        return None
    try:
        raw = ep_dialect_entry[seg_id]["dialect"]["vox_profile"][1]
        probs = _unwrap(raw)
        arr = np.array(probs, dtype=float)
        if arr.shape != (len(DIALECT_LABEL_LIST),):
            logging.warning(f"{seg_id}: unexpected dialect shape {arr.shape}, skipping")
            return None
        return arr
    except (KeyError, TypeError, ValueError) as e:
        logging.warning(f"{seg_id}: dialect extraction failed ({e}), skipping")
        return None


def extract_age_sex(ep_age_sex_entry, seg_id):
    """Return (age_years, female_prob, male_prob) for seg_id, or (None, None, None)."""
    if seg_id not in ep_age_sex_entry:
        return None, None, None
    try:
        entry = ep_age_sex_entry[seg_id]
        age = float(entry["age"]["vox_profile"])
        sex_probs = _unwrap(entry["sex"]["vox_profile"][1])
        if len(sex_probs) != 2:
            logging.warning(f"{seg_id}: unexpected sex probs len {len(sex_probs)}, skipping")
            return None, None, None
        return age, float(sex_probs[0]), float(sex_probs[1])
    except (KeyError, TypeError, ValueError) as e:
        logging.warning(f"{seg_id}: age/sex extraction failed ({e}), skipping")
        return None, None, None


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_file", default="sporc-single-speaker.jsonl")
    parser.add_argument("--filtered_output_dir", default="filtered_output")
    parser.add_argument("--utterance_labels_dir", default="utterance_labels")
    parser.add_argument("--output_file", default="sporc_speaker_info.json")
    parser.add_argument(
        "--transcript_key",
        default="canary_text",
        choices=["canary_text", "whisper_large_v3_text"],
        help="Which ASR transcript to use per segment (default: canary_text)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s ==> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. Merge utterance-label shards ----------------------------------
    utt_dir = Path(args.utterance_labels_dir)
    dialect_dict = merge_shards(str(utt_dir / "dialect*.json"))
    age_sex_dict = merge_shards(str(utt_dir / "age_sex*.json"))
    logging.info(
        f"Merged labels: {len(dialect_dict)} dialect episodes, "
        f"{len(age_sex_dict)} age_sex episodes"
    )

    # 2. Episode -> host-name(s) map from jsonl ------------------------
    ep_to_hosts = load_episode_to_hosts(args.jsonl_file)
    logging.info(f"Loaded {len(ep_to_hosts)} episodes from jsonl")

    # 3. Walk filtered_output/ and accumulate per-speaker stats --------
    # speaker -> {
    #   "episodes": {ep_id: [transcript1, ...]},
    #   "dialect_probs": [np.array(16,), ...],    # across all utterances
    #   "ages":           [float, ...],
    #   "female_probs":   [float, ...],
    #   "male_probs":     [float, ...],
    # }
    filter_dir = Path(args.filtered_output_dir)
    speakers = defaultdict(lambda: {
        "episodes": defaultdict(list),
        "dialect_probs": [],
        "ages": [],
        "female_probs": [],
        "male_probs": [],
    })

    skipped_empty = skipped_no_host = skipped_multi_host = 0

    for ep_file in sorted(filter_dir.glob("ep_*.json")):
        ep_id = ep_file.stem
        with open(ep_file) as f:
            segments = json.load(f)

        if not segments:
            skipped_empty += 1
            continue

        host_names = ep_to_hosts.get(ep_id, [])
        if not host_names:
            skipped_no_host += 1
            logging.warning(f"{ep_id}: no hostPredictedNames in jsonl, skipping")
            continue
        if len(host_names) > 1:
            skipped_multi_host += 1
            logging.warning(
                f"{ep_id}: {len(host_names)} host names, skipping episode"
            )
            continue
        speaker_id = host_names[0]

        ep_dialect = dialect_dict.get(ep_id, {})
        ep_age_sex = age_sex_dict.get(ep_id, {})

        sp = speakers[speaker_id]
        for seg_id, seg in segments.items():
            # Transcript is always kept (filter already gated agreement ≥90%).
            sp["episodes"][ep_id].append(seg.get(args.transcript_key, ""))

            dp = extract_dialect_probs(ep_dialect, seg_id)
            if dp is not None:
                sp["dialect_probs"].append(dp)

            age, fprob, mprob = extract_age_sex(ep_age_sex, seg_id)
            if age is not None:
                sp["ages"].append(age)
                sp["female_probs"].append(fprob)
                sp["male_probs"].append(mprob)

    logging.info(
        f"Walked filter outputs: {skipped_empty} empty, "
        f"{skipped_no_host} missing host, {skipped_multi_host} multi-host"
    )

    # 4. Build the final per-speaker dict ------------------------------
    out = {}
    for speaker_id in sorted(speakers.keys()):
        sp = speakers[speaker_id]
        info = {}

        # Episodes first (sorted for determinism).
        for ep_id in sorted(sp["episodes"].keys()):
            info[ep_id] = sp["episodes"][ep_id]

        # Age
        info["age"] = (
            round(float(np.mean(sp["ages"])), 1) if sp["ages"] else None
        )

        # Gender (argmax of mean [female, male] probability vector)
        if sp["female_probs"] and sp["male_probs"]:
            mean_f = float(np.mean(sp["female_probs"]))
            mean_m = float(np.mean(sp["male_probs"]))
            info["gender"] = SEX_LABEL_LIST[int(np.argmax([mean_f, mean_m]))]
        else:
            info["gender"] = "Unknown"

        # Accent (argmax of mean dialect probability vector) + diagnostics
        if sp["dialect_probs"]:
            avg_probs = np.mean(np.stack(sp["dialect_probs"]), axis=0)
            info["accent"] = DIALECT_LABEL_LIST[int(np.argmax(avg_probs))]
            info["accent_probs"] = {
                label: round(float(p), 4)
                for label, p in zip(DIALECT_LABEL_LIST, avg_probs)
            }
            info["accent_num_utterances"] = len(sp["dialect_probs"])
        else:
            info["accent"] = "Unknown"
            info["accent_probs"] = {}
            info["accent_num_utterances"] = 0

        out[speaker_id] = info

    # 5. Write output --------------------------------------------------
    with open(args.output_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # 6. Summary -------------------------------------------------------
    logging.info("-" * 60)
    logging.info(f"Output: {args.output_file}")
    logging.info(f"Total speakers:             {len(out)}")
    genders = [s["gender"] for s in out.values()]
    logging.info(
        f"Gender distribution:        "
        f"{genders.count('female')} female, "
        f"{genders.count('male')} male, "
        f"{genders.count('Unknown')} unknown"
    )
    ages = [s["age"] for s in out.values() if s["age"] is not None]
    if ages:
        logging.info(
            f"Age (across speakers):      mean={np.mean(ages):.1f}, "
            f"median={np.median(ages):.1f}, min={min(ages):.1f}, max={max(ages):.1f}"
        )
    with_accent = sum(1 for s in out.values() if s["accent"] != "Unknown")
    logging.info(f"Speakers with accent:       {with_accent}/{len(out)}")


if __name__ == "__main__":
    main()
