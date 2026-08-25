#!/usr/bin/env python3
"""Build cross-show, topic-controlled SPoRC trials from existing ASR data."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import html
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


META_KEYS = {"age", "gender", "accent", "accent_probs", "accent_num_utterances"}
SETTINGS = ("50_L5", "50_L20", "200_L5", "200_L20")
TASKS = ("speaker", "gender", "age", "accent")
TOPIC_EDGES = (0.005, 0.01, 0.025, 0.05, 0.075, 0.100001)
CATEGORY_EDGES = (0.001, 0.251, 0.501, 0.751, 1.001)
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Episode:
    episode_id: str
    speaker: str
    utterances: tuple[str, ...]
    gender: str
    age_group: str
    accent_group: str
    rss_url: str
    pod_title: str
    topic_text: str
    categories: frozenset[str]


@dataclass(frozen=True)
class Pair:
    probe: str
    enroll: str
    topic_similarity: float
    category_similarity: float


def age_to_group(age) -> str:
    try:
        value = float(age)
    except (TypeError, ValueError):
        return "Unknown"
    if value < 30:
        return "<30"
    if value < 60:
        return "30-60"
    return "60+"


def accent_to_group(accent) -> str:
    if not accent or str(accent).strip() == "Unknown":
        return "Unknown"
    return "North America" if str(accent).strip() == "North America" else "Other"


def clean_metadata_text(title: str, description: str) -> str:
    description = re.sub(r"<[^>]+>", " ", description or "")
    text = html.unescape(f"{title or ''} {description}")
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def utterance_set(utterances: tuple[str, ...]) -> set[str]:
    return {normalize_text(value) for value in utterances if normalize_text(value)}


def collect_probe(utterances: tuple[str, ...], target_words: int, tolerance: int) -> tuple[str, ...]:
    selected = []
    total = 0
    for utterance in utterances:
        selected.append(utterance)
        total += len(utterance.split())
        if total >= target_words - tolerance:
            return tuple(selected) if total <= target_words + tolerance else ()
    return ()


def category_jaccard(left: frozenset[str], right: frozenset[str]) -> float | None:
    union = left | right
    return len(left & right) / len(union) if union else None


def stratum(topic_similarity: float, category_similarity: float) -> tuple[int, int]:
    return (
        bisect.bisect_right(TOPIC_EDGES, topic_similarity),
        bisect.bisect_right(CATEGORY_EDGES, category_similarity),
    )


def parse_setting(setting: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)_L(\d+)", setting)
    if not match:
        raise ValueError(f"Invalid setting: {setting}")
    return int(match.group(1)), int(match.group(2))


def stable_id(setting: str, task: str, split: str, pair: Pair) -> str:
    payload = "\0".join((setting, task, split, pair.probe, pair.enroll))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{split}_tc_{digest}"


def load_raw_metadata(path: Path) -> dict[str, dict]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            record = json.loads(line)
            categories = frozenset(
                str(record.get(f"category{position}") or "").strip().lower()
                for position in range(1, 11)
                if str(record.get(f"category{position}") or "").strip()
            )
            result[f"ep_{index:06d}"] = {
                "rss_url": record.get("rssUrl") or "",
                "pod_title": record.get("podTitle") or "",
                "topic_text": clean_metadata_text(
                    record.get("epTitle") or "", record.get("epDescription") or ""
                ),
                "categories": categories,
            }
    return result


def load_episodes(speaker_info_path: Path, raw_metadata: dict[str, dict]) -> dict[str, Episode]:
    with speaker_info_path.open(encoding="utf-8") as handle:
        speaker_info = json.load(handle)

    episodes = {}
    for speaker, values in speaker_info.items():
        gender = str(values.get("gender") or "Unknown").lower().strip()
        age_group = age_to_group(values.get("age"))
        accent_group = accent_to_group(values.get("accent"))
        for episode_id, utterances in values.items():
            if episode_id in META_KEYS or episode_id not in raw_metadata:
                continue
            metadata = raw_metadata[episode_id]
            episodes[episode_id] = Episode(
                episode_id=episode_id,
                speaker=speaker,
                utterances=tuple(utterances),
                gender=gender,
                age_group=age_group,
                accent_group=accent_group,
                rss_url=metadata["rss_url"],
                pod_title=metadata["pod_title"],
                topic_text=metadata["topic_text"],
                categories=metadata["categories"],
            )
    return episodes


def fit_topic_model(episodes: dict[str, Episode]):
    episode_ids = sorted(episodes)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=100_000,
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform([episodes[key].topic_text for key in episode_ids])
    return matrix, {episode_id: index for index, episode_id in enumerate(episode_ids)}, vectorizer


def load_existing(existing_root: Path, setting: str) -> dict[tuple[str, str, str, str], dict]:
    result = {}
    for task in TASKS:
        for split in ("hard_positives", "easy_negatives"):
            path = existing_root / setting / task / f"{split}.csv"
            if not path.exists():
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    key = (task, split, row["probe_podcast"], row["enroll_podcast"])
                    result[key] = row
    return result


def task_label(task: str, probe: Episode, enroll: Episode) -> int:
    if task == "speaker":
        return int(probe.speaker == enroll.speaker)
    attribute = {
        "gender": "gender",
        "age": "age_group",
        "accent": "accent_group",
    }[task]
    probe_value = getattr(probe, attribute)
    enroll_value = getattr(enroll, attribute)
    if "Unknown" in (probe_value, enroll_value):
        return -1
    return int(probe_value == enroll_value)


def topic_similarity(matrix, indices: dict[str, int], left: str, right: str) -> float:
    return float(matrix[indices[left]].multiply(matrix[indices[right]]).sum())


def build_pair(
    left: Episode,
    right: Episode,
    matrix,
    indices: dict[str, int],
    threshold: float,
    require_cross_rss: bool = True,
) -> Pair | None:
    if not left.rss_url or not right.rss_url:
        return None
    if require_cross_rss and left.rss_url == right.rss_url:
        return None
    if not left.categories or not right.categories:
        return None
    similarity = topic_similarity(matrix, indices, left.episode_id, right.episode_id)
    if similarity > threshold:
        return None
    category_similarity = category_jaccard(left.categories, right.categories)
    if category_similarity is None:
        return None
    return Pair(left.episode_id, right.episode_id, similarity, category_similarity)


def leaks_identity(probe: Episode, enroll: Episode, probe_utts: tuple[str, ...], enroll_utts: tuple[str, ...]) -> bool:
    if utterance_set(probe_utts) & utterance_set(enroll_utts):
        return True
    speaker_name = normalize_text(probe.speaker)
    if len(speaker_name) < 4:
        return False
    probe_text = normalize_text(" ".join(probe_utts))
    enroll_text = normalize_text(" ".join(enroll_utts))
    return speaker_name in probe_text and speaker_name in enroll_text


def pair_key(pair: Pair) -> tuple[str, str]:
    return pair.probe, pair.enroll


def make_row(
    setting: str,
    task: str,
    split: str,
    pair: Pair,
    episodes: dict[str, Episode],
    probe_utts: dict[str, tuple[str, ...]],
    enroll_utts: dict[str, tuple[str, ...]],
    existing: dict[tuple[str, str, str, str], dict],
) -> dict:
    probe = episodes[pair.probe]
    enroll = episodes[pair.enroll]
    old = existing.get((task, split, pair.probe, pair.enroll))
    return {
        "trial_id": old["trial_id"] if old else stable_id(setting, task, split, pair),
        "label": 1 if split == "hard_positives" else 0,
        "probe_speaker": probe.speaker,
        "probe_podcast": probe.episode_id,
        "probe_utterances": " | ".join(probe_utts[probe.episode_id]),
        "probe_gender": probe.gender,
        "probe_age_group": probe.age_group,
        "probe_accent_group": probe.accent_group,
        "enroll_speaker": enroll.speaker,
        "enroll_podcast": enroll.episode_id,
        "enroll_utterances": " | ".join(enroll_utts[enroll.episode_id]),
        "enroll_gender": enroll.gender,
        "enroll_age_group": enroll.age_group,
        "enroll_accent_group": enroll.accent_group,
        "probe_rss_url": probe.rss_url,
        "enroll_rss_url": enroll.rss_url,
        "rss_relation": "same" if probe.rss_url == enroll.rss_url else "different",
        "probe_pod_title": probe.pod_title,
        "enroll_pod_title": enroll.pod_title,
        "probe_categories": " | ".join(sorted(probe.categories)),
        "enroll_categories": " | ".join(sorted(enroll.categories)),
        "topic_similarity": f"{pair.topic_similarity:.8f}",
        "category_similarity": f"{pair.category_similarity:.8f}",
        "pair_source": "retained" if old else "generated",
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty trial file: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_speaker_positives(
    episodes: dict[str, Episode],
    probe_utts: dict[str, tuple[str, ...]],
    enroll_utts: dict[str, tuple[str, ...]],
    matrix,
    indices: dict[str, int],
    threshold: float,
    require_cross_rss: bool,
    existing: dict[tuple[str, str, str, str], dict],
    max_generated_per_speaker: int,
    max_generated_per_probe: int,
    seed: int,
) -> list[Pair]:
    probes_by_speaker = defaultdict(list)
    enrollments_by_speaker = defaultdict(list)
    for episode in episodes.values():
        if episode.episode_id in probe_utts:
            probes_by_speaker[episode.speaker].append(episode)
        if episode.episode_id in enroll_utts:
            enrollments_by_speaker[episode.speaker].append(episode)

    rng = random.Random(seed)
    selected = []
    for speaker in sorted(probes_by_speaker):
        candidates = []
        for probe in probes_by_speaker[speaker]:
            for enroll in enrollments_by_speaker[speaker]:
                if probe.episode_id == enroll.episode_id:
                    continue
                pair = build_pair(
                    probe,
                    enroll,
                    matrix,
                    indices,
                    threshold,
                    require_cross_rss=require_cross_rss,
                )
                if pair is None:
                    continue
                if leaks_identity(
                    probe, enroll, probe_utts[probe.episode_id], enroll_utts[enroll.episode_id]
                ):
                    continue
                retained = ("speaker", "hard_positives", pair.probe, pair.enroll) in existing
                candidates.append((not retained, pair.topic_similarity, rng.random(), pair))
        candidates.sort()
        retained_pairs = [item[-1] for item in candidates if not item[0]]
        selected.extend(retained_pairs)
        seen = {pair_key(pair) for pair in retained_pairs}
        generated = 0
        generated_per_probe = Counter()
        for _, _, _, pair in candidates:
            if (
                pair_key(pair) in seen
                or generated_per_probe[pair.probe] >= max_generated_per_probe
            ):
                continue
            selected.append(pair)
            seen.add(pair_key(pair))
            generated_per_probe[pair.probe] += 1
            generated += 1
            if generated >= max_generated_per_speaker:
                break
    return selected


def build_cross_speaker_pool(
    episodes: dict[str, Episode],
    probe_utts: dict[str, tuple[str, ...]],
    enroll_utts: dict[str, tuple[str, ...]],
    matrix,
    indices: dict[str, int],
    threshold: float,
    existing: dict[tuple[str, str, str, str], dict],
    pool_size: int,
    seed: int,
) -> list[Pair]:
    probes = [episodes[key] for key in sorted(probe_utts)]
    enrollments = [episodes[key] for key in sorted(enroll_utts)]
    by_category = defaultdict(list)
    by_category_set = defaultdict(list)
    for episode in enrollments:
        by_category_set[episode.categories].append(episode)
        for category in episode.categories:
            by_category[category].append(episode)

    result = {}
    existing_episode_pairs = {
        (probe, enroll)
        for _, split, probe, enroll in existing
        if split in ("hard_positives", "easy_negatives")
    }
    for probe_id, enroll_id in sorted(existing_episode_pairs):
        if probe_id not in probe_utts or enroll_id not in enroll_utts:
            continue
        probe = episodes[probe_id]
        enroll = episodes[enroll_id]
        if probe.speaker == enroll.speaker:
            continue
        pair = build_pair(probe, enroll, matrix, indices, threshold)
        if pair is not None and not (utterance_set(probe_utts[probe_id]) & utterance_set(enroll_utts[enroll_id])):
            result[pair_key(pair)] = pair

    rng = random.Random(seed)
    attempts = 0
    max_attempts = pool_size * 80
    while len(result) < pool_size and attempts < max_attempts:
        attempts += 1
        probe = rng.choice(probes)
        enroll = None
        draw = rng.random()
        if probe.categories and draw < 0.70:
            candidates = by_category_set.get(probe.categories, ())
            if candidates:
                enroll = rng.choice(candidates)
        elif probe.categories and draw < 0.90:
            category = rng.choice(sorted(probe.categories))
            candidates = by_category.get(category, ())
            if candidates:
                enroll = rng.choice(candidates)
        if enroll is None:
            enroll = rng.choice(enrollments)
        if probe.speaker == enroll.speaker or probe.rss_url == enroll.rss_url:
            continue
        pair = build_pair(probe, enroll, matrix, indices, threshold)
        if pair is None or pair_key(pair) in result:
            continue
        if utterance_set(probe_utts[probe.episode_id]) & utterance_set(enroll_utts[enroll.episode_id]):
            continue
        result[pair_key(pair)] = pair

    if len(result) < pool_size:
        raise RuntimeError(f"Generated only {len(result):,}/{pool_size:,} cross-speaker pairs")
    return list(result.values())


def prioritize_existing(
    pairs: list[Pair],
    task: str,
    split: str,
    existing: dict[tuple[str, str, str, str], dict],
    seed: int,
) -> list[Pair]:
    rng = random.Random(seed)
    return sorted(
        pairs,
        key=lambda pair: (
            (task, split, pair.probe, pair.enroll) not in existing,
            rng.random(),
        ),
    )


def select_speaker_negatives(
    positives: list[Pair],
    pool: list[Pair],
    episodes: dict[str, Episode],
    existing: dict[tuple[str, str, str, str], dict],
    negatives_per_positive: int,
    seed: int,
) -> list[Pair]:
    eligible = []
    for pair in pool:
        probe = episodes[pair.probe]
        enroll = episodes[pair.enroll]
        if (
            probe.gender == enroll.gender != "Unknown"
            and probe.age_group == enroll.age_group != "Unknown"
            and probe.accent_group == enroll.accent_group != "Unknown"
        ):
            eligible.append(pair)

    grouped = defaultdict(list)
    for pair in eligible:
        grouped[stratum(pair.topic_similarity, pair.category_similarity)].append(pair)
    for key in grouped:
        grouped[key] = prioritize_existing(grouped[key], "speaker", "easy_negatives", existing, seed + sum(key))

    targets = Counter(stratum(pair.topic_similarity, pair.category_similarity) for pair in positives)
    selected = []
    for key, count in sorted(targets.items()):
        need = count * negatives_per_positive
        if len(grouped[key]) < need:
            raise RuntimeError(
                f"Speaker negative shortage in stratum {key}: need {need}, have {len(grouped[key])}"
            )
        selected.extend(grouped[key][:need])
    return selected


def proportional_allocations(counts: dict[tuple[int, int], int], target: int) -> dict[tuple[int, int], int]:
    total = sum(counts.values())
    if total <= target:
        return dict(counts)
    raw = {key: target * value / total for key, value in counts.items()}
    allocations = {key: int(value) for key, value in raw.items()}
    remaining = target - sum(allocations.values())
    for key in sorted(raw, key=lambda value: raw[value] - allocations[value], reverse=True)[:remaining]:
        allocations[key] += 1
    return allocations


def select_demographic_trials(
    task: str,
    pool: list[Pair],
    episodes: dict[str, Episode],
    existing: dict[tuple[str, str, str, str], dict],
    target_per_class: int,
    seed: int,
) -> tuple[list[Pair], list[Pair]]:
    grouped = {1: defaultdict(list), 0: defaultdict(list)}
    for pair in pool:
        label = task_label(task, episodes[pair.probe], episodes[pair.enroll])
        if label not in (0, 1):
            continue
        grouped[label][stratum(pair.topic_similarity, pair.category_similarity)].append(pair)

    capacities = {
        key: min(len(grouped[1][key]), len(grouped[0][key]))
        for key in set(grouped[1]) | set(grouped[0])
    }
    capacities = {key: value for key, value in capacities.items() if value}
    allocations = proportional_allocations(capacities, target_per_class)

    result = {1: [], 0: []}
    for label, split in ((1, "hard_positives"), (0, "easy_negatives")):
        for key, count in sorted(allocations.items()):
            values = prioritize_existing(grouped[label][key], task, split, existing, seed + label + sum(key))
            result[label].extend(values[:count])
    return result[1], result[0]


def summarize_rows(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "retained": sum(row["pair_source"] == "retained" for row in rows),
        "generated": sum(row["pair_source"] == "generated" for row in rows),
        "unique_probe_speakers": len({row["probe_speaker"] for row in rows}),
        "same_rss_rows": sum(row["probe_rss_url"] == row["enroll_rss_url"] for row in rows),
        "cross_rss_rows": sum(row["probe_rss_url"] != row["enroll_rss_url"] for row in rows),
        "mean_topic_similarity": round(float(np.mean([float(row["topic_similarity"]) for row in rows])), 6),
        "mean_category_similarity": round(float(np.mean([float(row["category_similarity"]) for row in rows])), 6),
        "strata": dict(sorted(Counter(
            str(stratum(float(row["topic_similarity"]), float(row["category_similarity"])))
            for row in rows
        ).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_jsonl", type=Path, required=True)
    parser.add_argument("--speaker_info", type=Path, default=REPO_ROOT / "data_processing" / "sporc" / "data" / "speaker_info.json")
    parser.add_argument("--existing_root", type=Path, default=None,
                        help="Optional older trial root whose qualifying rows are prioritized")
    parser.add_argument(
        "--additional_existing_roots",
        type=Path,
        nargs="*",
        default=[],
        help="Additional trial roots whose qualifying rows and IDs should be retained.",
    )
    parser.add_argument("--output_root", type=Path, default=Path("sporc_trials_topic_controlled"))
    parser.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=SETTINGS)
    parser.add_argument("--topic_threshold", type=float, default=0.10)
    parser.add_argument(
        "--speaker_positive_rss_policy",
        choices=("any", "different"),
        default="any",
        help="Allow same-RSS, different-episode speaker positives or require different RSS feeds.",
    )
    parser.add_argument("--tolerance", type=int, default=10)
    parser.add_argument("--cross_pool_size", type=int, default=80_000)
    parser.add_argument("--max_generated_positives_per_speaker", type=int, default=6)
    parser.add_argument("--max_generated_positives_per_probe", type=int, default=2)
    parser.add_argument("--speaker_negatives_per_positive", type=int, default=1)
    parser.add_argument("--demographic_trials_per_class", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_metadata = load_raw_metadata(args.raw_jsonl)
    episodes = load_episodes(args.speaker_info, raw_metadata)
    matrix, indices, vectorizer = fit_topic_model(episodes)
    print(f"Loaded {len(episodes):,} episodes; topic vocabulary={len(vectorizer.vocabulary_):,}")

    report = {
        "protocol": {
            "speaker_positive_rss_policy": args.speaker_positive_rss_policy,
            "cross_speaker_rss_policy": "different",
            "topic_method": "TF-IDF cosine over cleaned epTitle + epDescription",
            "topic_threshold": args.topic_threshold,
            "category_method": "Jaccard over non-empty Apple Podcast categories",
            "speaker_negative_demographic_match": ["gender", "age_group", "accent_group"],
            "speaker_negatives_per_positive": args.speaker_negatives_per_positive,
            "max_generated_positives_per_speaker": args.max_generated_positives_per_speaker,
            "max_generated_positives_per_probe": args.max_generated_positives_per_probe,
            "exact_utterance_overlap_removed": True,
            "explicit_speaker_name_in_both_sides_removed_from_positives": True,
            "same_and_cross_rss_positive_counts_reported_separately": True,
        },
        "settings": {},
    }

    for setting_index, setting in enumerate(args.settings):
        probe_words, enrollment_length = parse_setting(setting)
        probe_utts = {
            key: value
            for key, episode in episodes.items()
            if (value := collect_probe(episode.utterances, probe_words, args.tolerance))
        }
        enroll_utts = {
            key: tuple(episode.utterances[:enrollment_length])
            for key, episode in episodes.items()
            if episode.utterances[:enrollment_length]
        }
        existing = load_existing(args.existing_root, setting) if args.existing_root else {}
        for root in args.additional_existing_roots:
            existing.update(load_existing(root, setting))
        print(f"\n[{setting}] valid probes={len(probe_utts):,}, enrollments={len(enroll_utts):,}")

        positives = select_speaker_positives(
            episodes,
            probe_utts,
            enroll_utts,
            matrix,
            indices,
            args.topic_threshold,
            args.speaker_positive_rss_policy == "different",
            existing,
            args.max_generated_positives_per_speaker,
            args.max_generated_positives_per_probe,
            args.seed + setting_index,
        )
        cross_pool = build_cross_speaker_pool(
            episodes,
            probe_utts,
            enroll_utts,
            matrix,
            indices,
            args.topic_threshold,
            existing,
            args.cross_pool_size,
            args.seed + 100 + setting_index,
        )
        negatives = select_speaker_negatives(
            positives,
            cross_pool,
            episodes,
            existing,
            args.speaker_negatives_per_positive,
            args.seed + 200 + setting_index,
        )

        setting_report = {}
        for task in TASKS:
            if task == "speaker":
                positive_pairs, negative_pairs = positives, negatives
            else:
                positive_pairs, negative_pairs = select_demographic_trials(
                    task,
                    cross_pool,
                    episodes,
                    existing,
                    args.demographic_trials_per_class,
                    args.seed + 300 + setting_index,
                )

            task_report = {}
            for split, pairs in (("hard_positives", positive_pairs), ("easy_negatives", negative_pairs)):
                rows = [
                    make_row(
                        setting,
                        task,
                        split,
                        pair,
                        episodes,
                        probe_utts,
                        enroll_utts,
                        existing,
                    )
                    for pair in pairs
                ]
                write_rows(args.output_root / setting / task / f"{split}.csv", rows)
                task_report[split] = summarize_rows(rows)
                print(
                    f"  {task:7s} {split:15s}: {len(rows):,} "
                    f"(retained={task_report[split]['retained']:,})"
                )
            if task != "speaker" and task_report["hard_positives"]["strata"] != task_report["easy_negatives"]["strata"]:
                raise RuntimeError(f"Topic/category strata mismatch for {setting}/{task}")
            setting_report[task] = task_report
        report["settings"][setting] = setting_report

    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "generation_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nSaved generation report to {report_path}")


if __name__ == "__main__":
    main()
