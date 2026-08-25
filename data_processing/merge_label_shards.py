#!/usr/bin/env python3
"""Merge disjoint per-shard JSON dictionaries into one JSON file."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", required=True,
                        help="Quoted glob matching shard JSON files")
    parser.add_argument("--output_file", type=Path, required=True)
    args = parser.parse_args()

    paths = [Path(path) for path in sorted(glob.glob(args.input_glob))]
    if not paths:
        parser.error(f"No files matched: {args.input_glob}")

    merged: dict = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            shard = json.load(handle)
        overlap = merged.keys() & shard.keys()
        if overlap:
            sample = ", ".join(sorted(overlap)[:3])
            raise ValueError(f"Duplicate keys while reading {path}: {sample}")
        merged.update(shard)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2)
    print(f"Merged {len(paths)} shards and {len(merged):,} records into {args.output_file}")


if __name__ == "__main__":
    main()
