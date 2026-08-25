"""
SPoRC age + sex inference.

Uses a Vox-Profile WavLM checkpoint to predict speaker sex
(2-class: Female/Male) and age (regressed in [0, 1], multiplied by 100 to
get years) on the utterances that survived SPoRC filtering.

Mirrors the structure of infer_dialect_sporc.py so both scripts can share
the same input dir (filtered_output/) and output dir (utterance_labels/).

-------------------------------------------------------------------------------
Model interface (confirmed from working run_age_sex_classification.py)
-------------------------------------------------------------------------------
    # from vox-profile-release/src/model/age_sex/
    from wavlm_demographics import WavLMWrapper
    model = WavLMWrapper.from_pretrained(args.model_id)
    age_pred, sex_logits = model(audio)       # audio: [1, T] mono 16 kHz

The import below assumes the vox-profile-release source tree is importable as
`vox_profile.src.model.age_sex.wavlm_demographics`. If your environment puts
that directory on sys.path differently, adjust the import line only.
-------------------------------------------------------------------------------

Usage:
  # Single GPU:
  python infer_age_sex_sporc.py \
      --filtered_output_dir filtered_output \
      --audio_dir . \
      --output_dir utterance_labels

  # 4 GPUs in parallel (SLURM array of 4 tasks):
  python infer_age_sex_sporc.py --shard_id 0 --num_shards 4
  python infer_age_sex_sporc.py --shard_id 1 --num_shards 4
  python infer_age_sex_sporc.py --shard_id 2 --num_shards 4
  python infer_age_sex_sporc.py --shard_id 3 --num_shards 4
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
import soundfile as sf
from tqdm import tqdm

# Match the directory layout used by infer_dialect_sporc.py
sys.path.append(str(Path(os.path.realpath(__file__)).parents[1]))
sys.path.append(str(Path(os.path.realpath(__file__)).parents[1] / "model" / "age_sex"))

# Note: we do NOT call torchaudio.set_audio_backend() — it's deprecated in
# newer torchaudio and load_audio() below bypasses torchaudio.load() entirely.

logging.basicConfig(
    format="%(asctime)s %(levelname)-3s ==> %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

SAMPLE_RATE = 16000

# Vox-Profile: sex is 2-class, order given in the README as Female, Male.
SEX_LABEL_LIST = ["Female", "Male"]


# ============================================================================
# Robust audio loading (identical to infer_dialect_sporc.py / filter script)
# ============================================================================
def load_audio(audio_path: str, target_sr: int = 16000):
    """
    Load an audio file and return (waveform_tensor, sample_rate) as
    (channels, samples) float32. Bypasses torchaudio.load() to avoid its
    backend-registration system (brittle across version combinations).

    Strategy:
      1. soundfile directly — wav/flac/ogg, plus mp3 if libsndfile >= 1.1.0.
      2. ffmpeg to temp wav + soundfile — m4a/aac/opus and older-libsndfile mp3.
    """
    # Try 1: soundfile directly.
    try:
        audio_np, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        return torch.from_numpy(audio_np.T).contiguous(), sr
    except Exception:
        pass

    # Try 2: ffmpeg to temp wav, then soundfile.
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-ac", "1",
                "-ar", str(target_sr),
                "-sample_fmt", "s16",
                "-loglevel", "error",
                tmp_wav.name,
            ],
            check=True,
            capture_output=True,
        )
        audio_np, sr = sf.read(tmp_wav.name, dtype="float32", always_2d=True)
        return torch.from_numpy(audio_np.T).contiguous(), sr
    except Exception as e:
        raise RuntimeError(
            f"Both soundfile and ffmpeg+soundfile failed on '{audio_path}': {e}"
        )
    finally:
        Path(tmp_wav.name).unlink(missing_ok=True)


def resolve_audio_path(audio_file_field: str, audio_dir: Path) -> Path:
    """Resolve the `audio_file` JSON field against audio_dir, tolerating absolute paths."""
    p = Path(audio_file_field)
    if p.is_absolute() and p.exists():
        return p
    candidate = audio_dir / audio_file_field
    if candidate.exists():
        return candidate
    if p.exists():
        return p
    return candidate


# ============================================================================
# Model output unpacking — ADJUST IF NEEDED
# ============================================================================
def run_age_sex(model, audio: torch.Tensor):
    """
    Run a single forward pass and return (sex_label, sex_probs, age_years).

    Model interface:
      age_pred, sex_logits = model(audio)

      audio:      [1, T] tensor, mono 16 kHz
      age_pred:   scalar-ish tensor in [0, 1]; years = age_pred * 100
      sex_logits: [1, 2] tensor of logits for [Female, Male]
    """
    with torch.no_grad():
        age_pred, sex_logits = model(audio)

    sex_probs = (
        F.softmax(sex_logits, dim=1)
        .detach().cpu().numpy().astype(float).tolist()
    )
    sex_label = SEX_LABEL_LIST[int(np.argmax(sex_probs))]

    # age_pred is a 1-element tensor (shape [1] or []); handle both robustly
    age_val = age_pred.detach().cpu().float().flatten()[0].item()
    age_years = float(age_val) * 100.0

    return sex_label, sex_probs, age_years


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
        help="Directory to write age_sex JSON(s) into (shared with dialect script)",
    )
    parser.add_argument(
        "--model_id",
        default=os.environ.get("AGE_SEX_MODEL_ID"),
        help="Age/sex checkpoint (or set AGE_SEX_MODEL_ID)",
    )
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()
    if not args.model_id:
        parser.error("--model_id or AGE_SEX_MODEL_ID is required")

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
    # The checkpoint implementation exposes WavLMWrapper.
    # If this import path does not match your environment, adjust it. The
    # working script uses:
    #   sys.path.insert(0, "<...>/vox-profile-release/src/model/age_sex")
    #   from wavlm_demographics import WavLMWrapper
    from vox_profile.src.model.age_sex.wavlm_demographics import WavLMWrapper

    model = WavLMWrapper.from_pretrained(args.model_id).to(device)
    model.eval()

    # --- Collect episode files and shard (interleaved, same as dialect script) ---
    split_files = sorted(f for f in os.listdir(filtered_dir)
                         if f.endswith(".json") and f.startswith("ep_"))
    my_files = split_files[args.shard_id::args.num_shards]
    logging.info(
        f"Shard {args.shard_id}/{args.num_shards}: "
        f"{len(my_files)} of {len(split_files)} episode files"
    )

    # --- Resume support: separate file from dialect, per-shard ---
    if args.num_shards == 1:
        out_json_path = output_dir / "age_sex.json"
    else:
        out_json_path = output_dir / f"age_sex.shard_{args.shard_id}_of_{args.num_shards}.json"

    if out_json_path.exists():
        with open(out_json_path, "r") as f:
            age_sex_dict = json.load(f)
        logging.info(f"Resuming: {len(age_sex_dict)} episodes already processed")
    else:
        age_sex_dict = {}

    # --- Main loop (structure mirrors infer_dialect_sporc.py) ---
    for file in my_files:
        file_id = file.replace(".json", "")
        if file_id in age_sex_dict:
            continue

        with open(filtered_dir / file, "r") as f:
            json_data = json.load(f)

        # Empty dict = "done marker" from the filter script; record and skip.
        if not json_data:
            age_sex_dict[file_id] = {}
            with open(out_json_path, "w") as f:
                json.dump(age_sex_dict, f, indent=4)
            continue

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
            full_audio = full_audio.mean(dim=0, keepdim=True)

        # --- Per-utterance inference ---
        age_sex_dict[file_id] = {}
        for utterance_id in tqdm(json_data, desc=file_id):
            seg = json_data[utterance_id]
            start_sample = int(seg["start"] * SAMPLE_RATE)
            end_sample = int(seg["end"] * SAMPLE_RATE)

            end_sample = min(end_sample, full_audio.shape[1])
            if end_sample <= start_sample:
                continue

            audio_data = full_audio[:, start_sample:end_sample].to(device)

            try:
                sex_label, sex_probs, age_years = run_age_sex(model, audio_data)
            except Exception as e:
                logging.warning(
                    f"{file_id}/{utterance_id}: inference failed ({e}), skipping utterance"
                )
                continue

            # Mirror the dialect script's {attribute: {source: value}} shape so
            # downstream aggregation (infer_accent) can treat all three the same way.
            age_sex_dict[file_id][utterance_id] = {
                "sex": {"vox_profile": [sex_label, sex_probs]},
                "age": {"vox_profile": round(age_years, 2)},
            }

        # Checkpoint after each episode.
        with open(out_json_path, "w") as f:
            json.dump(age_sex_dict, f, indent=4)
