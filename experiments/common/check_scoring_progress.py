#!/usr/bin/env python3
"""Report trial-scoring progress from trial CSVs and result JSONs.

Progress is counted as completed model/trial score cells:
    completed = rows whose result JSON has a non-null score for the model
    expected  = rows in the matching trial CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path


DEFAULT_CONDITIONS = ["50_L5", "50_L20", "200_L5", "200_L20"]
DEFAULT_TASKS = ["speaker", "gender", "age", "accent"]
DEFAULT_SPLITS = ["easy_positives", "hard_positives", "easy_negatives", "hard_negatives"]
DEFAULT_MODELS = ["luar", "sbert", "tfidf", "cisr", "pcm", "llama", "qwen", "gpt"]


def load_trial_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return []
        if "trial_id" in reader.fieldnames:
            return [row["trial_id"] for row in reader if row.get("trial_id")]
        return [str(i) for i, _ in enumerate(reader)]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    for _ in range(5):
        try:
            with path.open(encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError:
            time.sleep(0.2)
    return {}


def pct(done: int, total: int) -> float:
    return 100.0 * done / total if total else 0.0


def print_row(label: str, done: int, total: int, width: int = 26) -> None:
    print(f"{label:<{width}} {done:>10,} / {total:<10,} {pct(done, total):6.2f}%")


def count_model_states(trial_ids: list[str], scores: dict, model: str) -> dict[str, int]:
    states = {"complete": 0, "missing": 0, "null": 0}
    for trial_id in trial_ids:
        trial_scores = scores.get(trial_id)
        if not isinstance(trial_scores, dict) or model not in trial_scores:
            states["missing"] += 1
        elif trial_scores[model] is None:
            states["null"] += 1
        else:
            states["complete"] += 1
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description="Check trial-scoring progress.")
    parser.add_argument("--trial_root", default="trials")
    parser.add_argument("--result_root", default="results")
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show every condition/task/split/model with incomplete scores.",
    )
    args = parser.parse_args()

    trial_root = Path(args.trial_root)
    result_root = Path(args.result_root)

    total_done = 0
    total_expected = 0
    by_condition = {c: [0, 0] for c in args.conditions}
    by_model = {m: [0, 0] for m in args.models}
    missing_files: list[str] = []
    incomplete_details: list[tuple[str, str, str, str, int, int, int, int]] = []

    for condition in args.conditions:
        for task in args.tasks:
            scored_dir = result_root / condition / f"{task}_{condition}"
            for split in args.splits:
                csv_path = trial_root / condition / task / f"{split}.csv"
                json_path = scored_dir / f"{split}.json"
                if not csv_path.exists():
                    continue
                trial_ids = load_trial_ids(csv_path)
                n_rows = len(trial_ids)

                if n_rows == 0:
                    missing_files.append(str(csv_path))
                    continue

                scores = load_json(json_path)
                if not scores:
                    missing_files.append(str(json_path))

                for model in args.models:
                    expected = n_rows
                    states = count_model_states(trial_ids, scores, model)
                    done = states["complete"]

                    total_done += done
                    total_expected += expected
                    by_condition[condition][0] += done
                    by_condition[condition][1] += expected
                    by_model[model][0] += done
                    by_model[model][1] += expected

                    incomplete = expected - done
                    if args.details and incomplete:
                        incomplete_details.append((
                            condition,
                            task,
                            split,
                            model,
                            incomplete,
                            states["missing"],
                            states["null"],
                            expected,
                        ))

    print("\nOverall")
    print_row("all", total_done, total_expected)

    print("\nBy condition")
    for condition in args.conditions:
        print_row(condition, *by_condition[condition])

    print("\nBy model")
    for model in args.models:
        print_row(model, *by_model[model])

    if args.details and incomplete_details:
        print("\nIncomplete details")
        print(
            f"{'condition':<9} {'task':<8} {'split':<15} {'model':<8} "
            f"{'incomplete':>10} {'missing':>10} {'null':>10} {'expected':>10}"
        )
        for condition, task, split, model, incomplete, missing, null, expected in incomplete_details:
            print(
                f"{condition:<9} {task:<8} {split:<15} {model:<8} "
                f"{incomplete:>10,} {missing:>10,} {null:>10,} {expected:>10,}"
            )

    if missing_files:
        print("\nMissing/empty inputs noticed:")
        for path in missing_files[:20]:
            print(f"  {path}")
        if len(missing_files) > 20:
            print(f"  ... and {len(missing_files) - 20} more")


if __name__ == "__main__":
    main()
