"""
SPoRC dialect inference.

Adapted from the NaturalVoices infer_dialect.py. Key differences:
  - Input: per-episode JSONs in filtered_output/ (output of sporc_data_filtering.py),
    not split_canary_whisper/.
  - Audio: mp3/m4a files referenced by the `audio_file` field inside each segment,
    not FLACs derived from file_id. Uses an ffmpeg fallback for m4a/aac.
  - Sharding: --shard_id / --num_shards to run in parallel across GPUs; each shard
    writes its own dialect JSON to avoid write races.
  - Output format is kept identical to the NaturalVoices version so downstream
    per-speaker aggregation stays compatible.

Usage:
  # Single GPU:
  python infer_dialect_sporc.py \
      --filtered_output_dir filtered_output \
      --audio_dir . \
      --output_dir utterance_labels

  # 4 GPUs in parallel (e.g. a SLURM array of 4 tasks):
  python infer_dialect_sporc.py --shard_id 0 --num_shards 4
  python infer_dialect_sporc.py --shard_id 1 --num_shards 4
  python infer_dialect_sporc.py --shard_id 2 --num_shards 4
  python infer_dialect_sporc.py --shard_id 3 --num_shards 4
"""

import os
import sys
import json
import argparse
import logging
import tempfile
import subprocess
from pathlib import Path

import torch
import torchaudio
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

# Match directory layout of the NaturalVoices infer script for WhisperWrapper import
sys.path.append(str(Path(os.path.realpath(__file__)).parents[1]))
sys.path.append(str(Path(os.path.realpath(__file__)).parents[1] / "model" / "dialect"))

torchaudio.set_audio_backend("soundfile")

logging.basicConfig(
    format="%(asctime)s %(levelname)-3s ==> %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

SAMPLE_RATE = 16000

DIALECT_LABEL_LIST = [
    "Eastern Asia", "English", "Germanic", "Irish",
    "North America", "Northern Irish", "Oceania",
    "Other", "Romance", "Scottish", "Semitic", "Slavic",
    "South African", "South-eastern Asia", "Southern Asia", "Welsh",
]


# ============================================================================
# Robust audio loading (copied from sporc_data_filtering.py for consistency)
# ============================================================================
def load_audio(audio_path: str, target_sr: int = 16000):
    """
    Load an audio file and return (waveform_tensor, sample_rate).

    Tries torchaudio first (handles mp3/wav/flac via soundfile or sox backends).
    Falls back to ffmpeg for formats the soundfile backend can't decode
    (m4a, aac, opus, etc.), decoding to a temp wav.
    """
    try:
        audio_data, sr = torchaudio.load(audio_path)
        return audio_data, sr
    except Exception:
        pass

    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-ac", "1",             # mono
                "-ar", str(target_sr),  # resample
                "-sample_fmt", "s16",   # 16-bit PCM
                "-loglevel", "error",
                tmp_wav.name,
            ],
            check=True,
            capture_output=True,
        )
        audio_data, sr = torchaudio.load(tmp_wav.name)
        return audio_data, sr
    except Exception as e:
        raise RuntimeError(
            f"Both torchaudio and ffmpeg failed on '{audio_path}': {e}"
        )
    finally:
        Path(tmp_wav.name).unlink(missing_ok=True)


def resolve_audio_path(audio_file_field: str, audio_dir: Path) -> Path:
    """
    Resolve the `audio_file` field from a segment JSON to an actual filesystem path.

    The field is typically a relative path like "audio_files/002_audio.m4a", but
    the filter script may have written an absolute path in some runs. Handle both.
    """
    p = Path(audio_file_field)
    if p.is_absolute() and p.exists():
        return p
    candidate = audio_dir / audio_file_field
    if candidate.exists():
        return candidate
    # Last resort: check whether the relative path resolves from cwd
    if p.exists():
        return p
    # Return the most-likely intended path so the caller can log a good error
    return candidate


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filtered_output_dir",
        type=str,
        default="filtered_output",
        help="Directory of per-episode JSONs produced by sporc_data_filtering.py",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default=".",
        help=(
            "Base directory for audio files. The `audio_file` field in each segment "
            "JSON (e.g. 'audio_files/002_audio.m4a') is resolved relative to this."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="utterance_labels",
        help="Directory to write dialect JSON(s) into (shared with other inference scripts)",
    )
    parser.add_argument(
        "--model_id",
        default=os.environ.get("ACCENT_MODEL_ID"),
        help="Accent checkpoint (or set ACCENT_MODEL_ID)",
    )
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()
    if not args.model_id:
        parser.error("--model_id or ACCENT_MODEL_ID is required")

    assert 0 <= args.shard_id < args.num_shards, \
        f"shard_id {args.shard_id} out of range for num_shards {args.num_shards}"

    filtered_dir = Path(args.filtered_output_dir)
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Device ---
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        logging.info(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("No GPU found, running on CPU (this will be slow).")

    # --- Model ---
    from vox_profile.src.model.accent.whisper_accent import WhisperWrapper

    whisper_model = WhisperWrapper.from_pretrained(args.model_id).to(device)
    whisper_model.eval()

    # --- Collect episode files and shard ---
    split_files = sorted(f for f in os.listdir(filtered_dir)
                         if f.endswith(".json") and f.startswith("ep_"))
    # Interleaved sharding so all shards finish at similar times even if
    # episodes vary in length.
    my_files = split_files[args.shard_id::args.num_shards]
    logging.info(
        f"Shard {args.shard_id}/{args.num_shards}: "
        f"{len(my_files)} of {len(split_files)} episode files"
    )

    # --- Resume support: each shard has its own dialect file ---
    if args.num_shards == 1:
        dialect_json_path = output_dir / "dialect.json"
    else:
        dialect_json_path = output_dir / f"dialect.shard_{args.shard_id}_of_{args.num_shards}.json"

    if dialect_json_path.exists():
        with open(dialect_json_path, "r") as f:
            dialect_dict = json.load(f)
        logging.info(f"Resuming: {len(dialect_dict)} episodes already processed")
    else:
        dialect_dict = {}

    # --- Main loop ---
    for file in my_files:
        file_id = file.replace(".json", "")
        if file_id in dialect_dict:
            continue

        with open(filtered_dir / file, "r") as f:
            json_data = json.load(f)

        # Empty dict = "done marker" from the filter script (episode had no
        # surviving segments). Record it and move on.
        if not json_data:
            dialect_dict[file_id] = {}
            with open(dialect_json_path, "w") as f:
                json.dump(dialect_dict, f, indent=4)
            continue

        # All segments in one episode share the same source audio; grab it from
        # any segment (taking the first here).
        first_seg = next(iter(json_data.values()))
        audio_file_field = first_seg.get("audio_file")
        if not audio_file_field:
            logging.warning(f"{file_id}: no `audio_file` field, skipping")
            continue

        audio_path = resolve_audio_path(audio_file_field, audio_dir)
        if not audio_path.exists():
            logging.warning(f"{file_id}: audio not found at {audio_path}, skipping")
            continue

        # --- Load once per episode ---
        try:
            full_audio, sr = load_audio(str(audio_path), target_sr=SAMPLE_RATE)
        except Exception as e:
            logging.warning(f"{file_id}: error loading {audio_path}: {e}, skipping")
            continue

        if sr != SAMPLE_RATE:
            full_audio = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(full_audio)
        if full_audio.shape[0] > 1:
            full_audio = full_audio.mean(dim=0, keepdim=True)  # to mono

        # --- Per-utterance inference ---
        dialect_dict[file_id] = {}
        for utterance_id in tqdm(json_data, desc=file_id):
            seg = json_data[utterance_id]
            start_sample = int(seg["start"] * SAMPLE_RATE)
            end_sample = int(seg["end"] * SAMPLE_RATE)

            # Safety: clamp to available audio (Brouhaha timestamps occasionally
            # overshoot the file end by a frame or two).
            end_sample = min(end_sample, full_audio.shape[1])
            if end_sample <= start_sample:
                continue

            audio_data = full_audio[:, start_sample:end_sample].to(device)

            dialect_dict[file_id][utterance_id] = {"dialect": {}}

            with torch.no_grad():
                whisper_logits, _whisper_embeddings = whisper_model(
                    audio_data, return_feature=True
                )
                ensemble_logits = whisper_logits
                ensemble_prob = (
                    F.softmax(ensemble_logits, dim=1)
                    .detach().cpu().numpy().astype(float).tolist()
                )

            dialect_label = DIALECT_LABEL_LIST[int(np.argmax(ensemble_prob))]
            dialect_dict[file_id][utterance_id]["dialect"]["vox_profile"] = [
                dialect_label, ensemble_prob
            ]

        # Checkpoint after each episode so a crash doesn't lose a shard's work.
        with open(dialect_json_path, "w") as f:
            json.dump(dialect_dict, f, indent=4)

    logging.info(f"Done! Dialect results saved to {dialect_json_path}")