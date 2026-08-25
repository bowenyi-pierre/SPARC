"""
SPoRC Dual-ASR pipeline with AST music filtering and SNR estimation.
Adapted from the NaturalVoices data_filtering.py pipeline.

Key differences from NaturalVoices:
  - No license filtering (not available in SPoRC).
  - Full-episode audio → VAD + SNR via Brouhaha (single pass).
  - Audio files named by line index in urls.txt (e.g., 001_audio.mp3).
  - Whisper large-v3 instead of Whisper Turbo.
  - Metadata is JSONL (one JSON object per line), not nested JSON.

Filtering thresholds (matching NaturalVoices data_filtering.py):
  - SNR >= 15 dB
  - Music event probability <= 0.1  (AST sum over music categories)
  - Dual ASR agreement WER <= 0.1   (i.e., >90% agreement)

Usage:
  # Single GPU:
  python sporc_data_filtering.py --urls_file urls.txt --jsonl_file sporc-single-speaker.jsonl

  # 4 GPUs in parallel:
  python sporc_data_filtering.py --shard_id 0 --num_shards 4
  python sporc_data_filtering.py --shard_id 1 --num_shards 4
  python sporc_data_filtering.py --shard_id 2 --num_shards 4
  python sporc_data_filtering.py --shard_id 3 --num_shards 4
"""

import re
import sys
import json
import torch
import whisper
import tempfile
import argparse
import subprocess
import torchaudio
import numpy as np
import soundfile as sf
import torchaudio.functional as F

from jiwer import wer
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from nemo.collections.speechlm2.models import SALM
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from pyannote.audio import Model as PyannoteModel
from brouhaha.pipeline import RegressiveActivityDetectionPipeline
from huggingface_hub import hf_hub_download

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s',
    level=logging.WARNING,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def stage_log(message: str):
    """
    Emit explicit progress lines when stdout/stderr are redirected to logs.
    tqdm redraws are hard to interpret in files, so these messages make it
    obvious which episode/stage is currently running.
    """
    if sys.stderr.isatty():
        return
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} INFO ==> {message}", flush=True)

import torch

# --- PyTorch 2.6 Checkpoint Workaround ---
_original_load = torch.load

def _patched_load(*args, **kwargs):
    # Force weights_only to False to bypass the strict security filter
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

# Overwrite torch.load with our patched version
torch.load = _patched_load
# -----------------------------------------

# ============================================================================
# Thresholds — matched to NaturalVoices data_filtering.py
# ============================================================================
SAMPLE_RATE = 16000
SNR_THRESHOLD = 15            # dB — same as utt_info["SNR"] >= 15
MUSIC_THRESHOLD = 0.1         # AST music sum — same as MUSIC_THRESHOLD = 0.1
WER_THRESHOLD = 0.1           # dual-ASR agreement — same as er1 > 0.1
MIN_DURATION = 3.0            # seconds — same as duration > 3
MAX_DURATION = 15.0           # seconds — same as duration <= 15

# Brouhaha outputs SNR/C50 at 16000/270 ≈ 59.26 frames per second
BROUHAHA_RATIO = SAMPLE_RATE / 270

# Batch sizes (tuneable via CLI)
AST_BATCH_SIZE = 64
WHISPER_BATCH_SIZE = 64       # large-v3 is heavier than turbo; smaller default
CANARY_BATCH_SIZE = 64

# ============================================================================
# Music categories from AudioSet ontology (identical to NaturalVoices)
# ============================================================================
MUSIC_CATEGORIES = {
    "A capella", "Accordion", "Acoustic guitar", "Afrobeat", "Ambient music",
    "Angry music", "Background music", "Bagpipes", "Banjo", "Bass drum",
    "Bass guitar", "Beatboxing", "Bell", "Bluegrass", "Blues",
    "Bowed string instrument", "Brass instrument", "Carnatic music", "Cello",
    "Change ringing (campanology)", "Chant", "Child singing", "Chime", "Choir",
    "Chorus effect", "Christian music", "Christmas music", "Church bell",
    "Clarinet", "Classical music", "Country", "Cowbell", "Cymbal", "Dance music",
    "Didgeridoo", "Disco", "Distortion", "Double bass", "Drum", "Drum and bass",
    "Drum kit", "Drum machine", "Drum roll", "Dubstep", "Electric guitar",
    "Electric piano", "Electronic dance music", "Electronic music",
    "Electronic organ", "Electronic tuner", "Electronica", "Exciting music",
    "Female singing", "Flamenco", "Flute", "Folk music", "French horn", "Funk",
    "Funny music", "Glockenspiel", "Gong", "Gospel music", "Grunge", "Guitar",
    "Hammond organ", "Happy music", "Harmonic", "Harmonica", "Harp",
    "Harpsichord", "Heavy metal", "Hi-hat", "Hip hop music", "House music",
    "Independent music", "Jazz", "Jingle (music)", "Jingle bell",
    "Keyboard (musical)", "Lullaby", "Male singing", "Mallet percussion",
    "Mandolin", "Mantra", "Maraca", "Marimba, xylophone", "Middle Eastern music",
    "Music", "Music for children", "Music of Africa", "Music of Asia",
    "Music of Bollywood", "Music of Latin America", "Musical instrument",
    "New-age music", "Opera", "Orchestra", "Organ", "Percussion", "Piano",
    "Pizzicato", "Plucked string instrument", "Pop music", "Progressive rock",
    "Psychedelic rock", "Punk rock", "Rapping", "Rattle (instrument)", "Reggae",
    "Rhythm and blues", "Rimshot", "Rock and roll", "Rock music", "Sad music",
    "Salsa music", "Sampler", "Saxophone", "Scary music",
    "Scratching (performance technique)", "Shofar", "Singing", "Singing bowl",
    "Sitar", "Ska", "Snare drum", "Song", "Soul music", "Soundtrack music",
    "Steel guitar, slide guitar", "Steelpan", "String section", "Strum",
    "Swing music", "Synthetic singing", "Synthesizer", "Tabla", "Tambourine",
    "Tapping (guitar technique)", "Techno", "Tender music", "Theme music",
    "Theremin", "Timpani", "Traditional music", "Trance music", "Trombone",
    "Trumpet", "Tubular bells", "Tuning fork", "Ukulele", "Vibraphone",
    "Video game music", "Violin, fiddle", "Vocal music", "Wedding music",
    "Wind chime", "Wind instrument, woodwind instrument", "Wood block",
    "Yodeling", "Zither"
}


# ============================================================================
# URL → audio file mapping
# ============================================================================
def build_url_to_filepath(urls_file: str, audio_dir: str) -> dict:
    """
    Build a mapping from mp3url → local audio file path.

    download.sh names files as NNN_audio.{ext} where NNN is the 1-based line
    number (zero-padded to 3 digits) and ext is detected from the URL
    (default mp3).
    """
    url_to_path = {}
    with open(urls_file, "r") as f:
        for i, line in enumerate(f, start=1):
            url = line.strip()
            if not url or url.startswith("#"):
                continue

            # Detect extension from URL (same logic as download.sh)
            ext = "mp3"
            url_no_query = url.split("?")[0]
            for candidate_ext in ("m4a", "mp3", "wav", "ogg", "flac"):
                if url_no_query.endswith(f".{candidate_ext}"):
                    ext = candidate_ext
                    break

            filename = f"{i:03d}_audio.{ext}"
            filepath = Path(audio_dir) / filename
            url_to_path[url] = str(filepath)

    return url_to_path


# ============================================================================
# Robust audio loading (ffmpeg fallback for m4a / aac)
# ============================================================================
def load_audio(audio_path: str, target_sr: int = 16000):
    """
    Load an audio file and return (waveform_tensor, sample_rate).

    Tries torchaudio first (fast, handles mp3/wav/flac).
    Falls back to ffmpeg for formats that the soundfile backend can't
    decode (m4a, aac, opus, etc.).
    """
    try:
        audio_data, sr = torchaudio.load(audio_path)
        return audio_data, sr
    except Exception:
        pass

    # Fallback: use ffmpeg to decode to a temp wav file
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-ac", "1",              # mono
                "-ar", str(target_sr),   # resample
                "-sample_fmt", "s16",    # 16-bit PCM
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


# ============================================================================
# Brouhaha: combined VAD + SNR (replaces pyannote VAD + WADA-SNR)
# ============================================================================
def load_brouhaha(device, batch_size=32):
    """Load Brouhaha model for joint VAD + SNR + C50 estimation."""
    model = PyannoteModel.from_pretrained(
        Path(hf_hub_download(repo_id="ylacombe/brouhaha-best", filename="best.ckpt")),
        strict=False,
    )
    model.to(device)
    pipeline = RegressiveActivityDetectionPipeline(segmentation=model, batch_size=batch_size)
    pipeline.to(device)
    return pipeline


def brouhaha_vad_and_snr(audio_tensor, brouhaha_pipeline, device,
                         sr=16000, min_dur=3.0, max_dur=15.0, snr_threshold=15.0):
    """
    Run Brouhaha on a full episode waveform to get VAD segments and per-segment SNR.

    This replaces the two-step approach of pyannote VAD + WADA-SNR with a single
    Brouhaha pass that jointly produces VAD annotations and frame-level SNR.

    Args:
        audio_tensor: 1D tensor (num_samples,) of the full episode, mono, at `sr`.
        brouhaha_pipeline: loaded RegressiveActivityDetectionPipeline.
        device: torch device.
        sr: sample rate (must be 16000).
        min_dur: minimum segment duration in seconds.
        max_dur: maximum segment duration in seconds.
        snr_threshold: SNR threshold in dB; segments below this are dropped.

    Returns:
        candidates: list of (seg_id, seg_audio_tensor, seg_info_dict) for segments
                    that pass the SNR threshold.
        num_snr_filtered: count of segments dropped by SNR.
    """
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)  # (1, num_samples)

    res = brouhaha_pipeline(
        {"sample_rate": sr, "waveform": audio_tensor.float().to(device)}
    )

    snr_array = res["snr"]   # frame-level SNR, shape (num_frames,)
    c50_array = res["c50"]   # frame-level C50, shape (num_frames,)

    # ---- Extract VAD segments from Brouhaha annotation ----
    raw_segments = []
    for speech_turn, _ in res["annotation"].itertracks():
        start_sec = speech_turn.start
        end_sec = speech_turn.end
        duration = end_sec - start_sec

        if duration > max_dur:
            # Split long segments into max_dur chunks
            offset = start_sec
            while offset + min_dur <= end_sec:
                chunk_end = min(offset + max_dur, end_sec)
                if chunk_end - offset >= min_dur:
                    raw_segments.append({"start": offset, "end": chunk_end})
                offset += max_dur
        elif duration >= min_dur:
            raw_segments.append({"start": start_sec, "end": end_sec})

    # ---- Compute per-segment SNR from frame-level array ----
    candidates = []
    num_snr_filtered = 0

    for seg_i, seg_info in enumerate(raw_segments):
        # Extract segment audio
        start_sample = int(seg_info["start"] * sr)
        end_sample = int(seg_info["end"] * sr)
        seg_audio = audio_tensor[0, start_sample:end_sample]

        # Map segment boundaries to Brouhaha frame indices
        frame_start = int(seg_info["start"] * BROUHAHA_RATIO)
        frame_end = int(seg_info["end"] * BROUHAHA_RATIO)
        frame_end = min(frame_end, len(snr_array))  # clamp to array bounds

        if frame_start >= frame_end:
            num_snr_filtered += 1
            continue

        seg_snr = snr_array[frame_start:frame_end]
        seg_c50 = c50_array[frame_start:frame_end]

        # Exclude zero-padding frames (where both SNR and C50 are exactly 0.0)
        valid_mask = ~((seg_snr == 0.0) & (seg_c50 == 0.0))

        if valid_mask.any():
            snr_val = float(seg_snr[valid_mask].mean())
        else:
            # All frames are zero — treat as silence
            num_snr_filtered += 1
            continue

        if snr_val < snr_threshold:
            num_snr_filtered += 1
            continue

        seg_info["snr"] = round(snr_val, 2)
        seg_info["c50"] = round(float(seg_c50[valid_mask].mean()), 2)
        candidates.append((f"seg_{seg_i:04d}", seg_audio, seg_info))

    return candidates, num_snr_filtered


# ============================================================================
# Batched AST music detection
# ============================================================================
def batch_check_music(segments_np, ast_model, ast_extractor, music_idx_tensor,
                      device, sr=16000):
    """
    Batch AST inference on multiple segments. Returns list of music_sum scores.

    For segments > 10s, checks both the first and last 10s windows and takes
    the max score, so music that appears late in a segment is not missed.
    """
    target_len = 10 * sr

    # Flatten: list of (segment_index, 10s_chunk) pairs
    chunks = []
    for seg_idx, audio in enumerate(segments_np):
        if len(audio) <= target_len:
            padded = np.pad(audio, (0, max(0, target_len - len(audio))))
            chunks.append((seg_idx, padded))
        else:
            # First 10s and last 10s (overlapping is fine)
            chunks.append((seg_idx, audio[:target_len]))
            chunks.append((seg_idx, audio[-target_len:]))

    # Run AST in batches over all chunks
    chunk_scores = []
    for i in range(0, len(chunks), AST_BATCH_SIZE):
        batch = [c[1] for c in chunks[i:i + AST_BATCH_SIZE]]
        inputs = ast_extractor(batch, sampling_rate=sr, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = torch.sigmoid(ast_model(**inputs).logits)
        batch_music = probs[:, music_idx_tensor].sum(dim=1).cpu().tolist()
        chunk_scores.extend(batch_music)

    # Aggregate per-segment: take the max across chunks
    music_sums = [0.0] * len(segments_np)
    for (seg_idx, _), score in zip(chunks, chunk_scores):
        music_sums[seg_idx] = max(music_sums[seg_idx], score)

    return music_sums


# ============================================================================
# Batched Whisper large-v3 decode
# ============================================================================
def batch_whisper_decode(segments, whisper_model, device):
    """
    Batch Whisper mel computation + decode.
    Returns list of (language, text) tuples.
    """
    results = []
    for i in range(0, len(segments), WHISPER_BATCH_SIZE):
        batch = segments[i:i + WHISPER_BATCH_SIZE]
        mels = []
        for seg in batch:
            padded = whisper.pad_or_trim(seg)
            mel = whisper.log_mel_spectrogram(
                padded, n_mels=whisper_model.dims.n_mels
            )
            mels.append(mel)
        mel_batch = torch.stack(mels).to(device)
        options = whisper.DecodingOptions()
        decode_results = whisper.decode(whisper_model, mel_batch, options)
        for r in decode_results:
            results.append((r.language, r.text))
    return results


# ============================================================================
# Batched Canary transcription (identical to NaturalVoices)
# ============================================================================
def batch_canary_transcribe(canary_model, wav_paths, max_new_tokens=128):
    """
    Batch Canary SALM transcription.
    Returns list of transcribed text strings.
    """
    texts = []
    for i in range(0, len(wav_paths), CANARY_BATCH_SIZE):
        batch_paths = wav_paths[i:i + CANARY_BATCH_SIZE]
        prompts = [
            [{"role": "user",
              "content": f"Transcribe the following: {canary_model.audio_locator_tag}",
              "audio": [p]}]
            for p in batch_paths
        ]
        try:
            answer_ids = canary_model.generate(
                prompts=prompts, max_new_tokens=max_new_tokens
            )
            for ids in answer_ids:
                texts.append(canary_model.tokenizer.ids_to_text(ids.cpu()))
        except Exception:
            # Fallback: process one by one if batch fails
            for p in batch_paths:
                try:
                    ids = canary_model.generate(
                        prompts=[[{
                            "role": "user",
                            "content": f"Transcribe the following: {canary_model.audio_locator_tag}",
                            "audio": [p]
                        }]],
                        max_new_tokens=max_new_tokens,
                    )
                    texts.append(canary_model.tokenizer.ids_to_text(ids[0].cpu()))
                except Exception:
                    texts.append("")
    return texts


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="SPoRC data filtering: SNR + music + dual-ASR agreement"
    )
    # Paths
    parser.add_argument("--urls_file", type=str,
                        default="urls.txt",
                        help="Full urls.txt (all lines, not the 100-line sample)")
    parser.add_argument("--jsonl_file", type=str,
                        default="sporc-single-speaker.jsonl",
                        help="Full SPoRC JSONL metadata file")
    parser.add_argument("--audio_dir", type=str,
                        default="audio_files",
                        help="Directory containing downloaded audio files")
    parser.add_argument("--output_dir", type=str,
                        default="filtered_output",
                        help="Output directory for per-episode filtered JSON")

    # Sharding
    parser.add_argument("--shard_id", type=int, default=0,
                        help="This worker's shard index (0-based)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of parallel workers")

    # Data split (for running on subsets of episodes)
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start episode index (inclusive, 0-based into JSONL)")
    parser.add_argument("--end_idx", type=int, default=-1,
                        help="End episode index (exclusive). -1 means all remaining.")

    # Batch sizes
    parser.add_argument("--ast_batch", type=int, default=AST_BATCH_SIZE)
    parser.add_argument("--whisper_batch", type=int, default=WHISPER_BATCH_SIZE)
    parser.add_argument("--canary_batch", type=int, default=CANARY_BATCH_SIZE)

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    args = parse_args()
    AST_BATCH_SIZE = args.ast_batch
    WHISPER_BATCH_SIZE = args.whisper_batch
    CANARY_BATCH_SIZE = args.canary_batch

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ---- Build URL → file path mapping ----
    print("Building URL → audio file mapping...")
    url_to_path = build_url_to_filepath(args.urls_file, args.audio_dir)
    print(f"  Mapped {len(url_to_path)} URLs to audio files.")

    # ---- Load JSONL metadata ----
    print("Loading JSONL metadata...")
    episodes = []
    with open(args.jsonl_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    print(f"  Loaded {len(episodes)} episodes.")

    # ---- Apply data split (slice before matching to audio) ----
    end_idx = args.end_idx if args.end_idx != -1 else len(episodes)
    episodes_slice = list(enumerate(episodes))[args.start_idx:end_idx]
    print(f"  Data split: episodes [{args.start_idx}, {end_idx}) → "
          f"{len(episodes_slice)} episodes in this split.")

    # ---- Match episodes to audio files and filter already-processed ----
    episode_tasks = []
    for idx, ep in episodes_slice:
        mp3url = ep.get("mp3url", "")
        if mp3url not in url_to_path:
            continue
        audio_path = url_to_path[mp3url]
        if not Path(audio_path).exists():
            continue

        # Use a sanitized episode identifier for the output filename
        ep_id = f"ep_{idx:06d}"
        output_file = Path(args.output_dir) / f"{ep_id}.json"
        if output_file.exists():
            continue  # skip already completed

        episode_tasks.append((idx, ep, audio_path, ep_id))

    # Apply sharding
    episode_tasks = sorted(episode_tasks, key=lambda x: x[0])
    episode_tasks = episode_tasks[args.shard_id::args.num_shards]
    print(f"Shard {args.shard_id}/{args.num_shards}: "
          f"{len(episode_tasks)} episodes to process\n")

    if len(episode_tasks) == 0:
        print("Nothing to process. Exiting.")
        exit(0)

    # ---- Find device ----
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        print("GPU available, using GPU")

    # ---- Load models ----
    print("Loading Brouhaha (VAD + SNR)...")
    brouhaha_pipeline = load_brouhaha(device)

    print("Loading Whisper large-v3...")
    whisper_model = whisper.load_model("large-v3").to(device)

    print("Loading Canary SALM (canary-qwen-2.5b)...")
    canary_model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
    if device.type == "cuda":
        canary_model = canary_model.bfloat16()
    canary_model = canary_model.eval().to(device)

    print("Loading AST music detector...")
    ast_model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    ast_extractor = ASTFeatureExtractor.from_pretrained(ast_model_name)
    ast_model = ASTForAudioClassification.from_pretrained(ast_model_name).to(device)
    ast_model.eval()

    music_label_indices = []
    for idx_label, label in ast_model.config.id2label.items():
        if label in MUSIC_CATEGORIES:
            music_label_indices.append(int(idx_label))
    music_idx_tensor = torch.tensor(music_label_indices, device=device)

    print(f"All models loaded.  Music labels: {len(music_label_indices)}, "
          f"Batch sizes: AST={AST_BATCH_SIZE}, Whisper={WHISPER_BATCH_SIZE}, "
          f"Canary={CANARY_BATCH_SIZE}\n")

    # ---- Counters (restore from checkpoint if resuming) ----
    checkpoint_file = Path(args.output_dir) / f"checkpoint_shard{args.shard_id}.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            ckpt = json.load(f)
        total_segments_kept = ckpt.get("total_segments_kept", 0)
        total_duration_kept = ckpt.get("total_duration_kept", 0.0)
        total_snr_filtered = ckpt.get("total_snr_filtered", 0)
        total_music_filtered = ckpt.get("total_music_filtered", 0)
        total_wer_mismatch = ckpt.get("total_wer_mismatch", 0)
        total_non_english = ckpt.get("total_non_english", 0)
        print(f"Resumed from checkpoint: {total_segments_kept} segments kept, "
              f"{total_duration_kept/3600:.2f}h so far")
    else:
        total_segments_kept = 0
        total_duration_kept = 0.0
        total_snr_filtered = 0
        total_music_filtered = 0
        total_wer_mismatch = 0
        total_non_english = 0

    def save_checkpoint():
        with open(str(checkpoint_file), "w") as f:
            json.dump({
                "total_segments_kept": total_segments_kept,
                "total_duration_kept": total_duration_kept,
                "total_snr_filtered": total_snr_filtered,
                "total_music_filtered": total_music_filtered,
                "total_wer_mismatch": total_wer_mismatch,
                "total_non_english": total_non_english,
            }, f)

    # ---- Main loop ----
    def mark_episode_done(ep_id):
        """Write empty marker + save checkpoint so this episode is skipped on resume."""
        marker = Path(args.output_dir) / f"{ep_id}.json"
        if not marker.exists():
            with open(str(marker), "w") as f:
                json.dump({}, f)
        save_checkpoint()

    show_tqdm = sys.stderr.isatty()
    ep_pbar = tqdm(episode_tasks, desc="Episodes", unit="ep", disable=not show_tqdm)
    for ep_idx, ep_meta, audio_path, ep_id in ep_pbar:
        if show_tqdm:
            ep_pbar.set_postfix_str(
                f"kept={total_segments_kept} music={total_music_filtered} "
                f"snr={total_snr_filtered} hrs={total_duration_kept/3600:.2f}"
            )
        stage_log(
            f"Shard {args.shard_id}: starting {ep_id} "
            f"(episode_index={ep_idx}, audio={Path(audio_path).name}) | "
            f"totals kept={total_segments_kept} music={total_music_filtered} "
            f"snr={total_snr_filtered} hrs={total_duration_kept/3600:.2f}"
        )

        # ---- Load audio ----
        try:
            audio_data, sr = load_audio(audio_path, target_sr=SAMPLE_RATE)
        except Exception as e:
            logging.warning(f"Skipping corrupted file: {audio_path} | {e}")
            stage_log(f"Shard {args.shard_id}: {ep_id} failed to load audio; marking done")
            mark_episode_done(ep_id)
            continue

        if sr != SAMPLE_RATE:
            audio_data = F.resample(audio_data, sr, SAMPLE_RATE)
        if audio_data.shape[0] > 1:
            audio_data = audio_data.mean(dim=0, keepdim=True)

        audio_mono = audio_data[0]  # (num_samples,)

        # ==== Stage 0+1: Brouhaha VAD + SNR filtering (single pass) ====
        candidates, snr_dropped = brouhaha_vad_and_snr(
            audio_mono, brouhaha_pipeline, device,
            sr=SAMPLE_RATE, min_dur=MIN_DURATION, max_dur=MAX_DURATION,
            snr_threshold=SNR_THRESHOLD
        )
        total_snr_filtered += snr_dropped
        stage_log(
            f"Shard {args.shard_id}: {ep_id} after Brouhaha | "
            f"candidates={len(candidates)} snr_dropped={snr_dropped}"
        )

        if not candidates:
            stage_log(f"Shard {args.shard_id}: {ep_id} has no candidates after SNR; marking done")
            mark_episode_done(ep_id)
            continue

        # ==== Stage 2: Batched AST music filter ====
        seg_arrays = [seg.numpy() for _, seg, _ in candidates]
        music_scores = batch_check_music(
            seg_arrays, ast_model, ast_extractor, music_idx_tensor, device
        )

        after_music = []
        for (seg_id, seg, info), mscore in zip(candidates, music_scores):
            if mscore <= MUSIC_THRESHOLD:
                info["music_sum"] = round(mscore, 4)
                after_music.append((seg_id, seg, info))
            else:
                total_music_filtered += 1

        stage_log(
            f"Shard {args.shard_id}: {ep_id} after AST music filter | "
            f"kept={len(after_music)} music_dropped={len(candidates) - len(after_music)}"
        )
        if not after_music:
            stage_log(f"Shard {args.shard_id}: {ep_id} has no candidates after music filter; marking done")
            mark_episode_done(ep_id)
            continue

        # ==== Stage 3: Batched Whisper large-v3 (language + transcription) ====
        whisper_segments = [seg for _, seg, _ in after_music]
        whisper_results = batch_whisper_decode(whisper_segments, whisper_model, device)

        after_whisper = []
        for (seg_id, seg, info), (lang, wtext) in zip(after_music, whisper_results):
            if lang == "en":
                after_whisper.append((seg_id, seg, info, wtext))
            else:
                total_non_english += 1

        stage_log(
            f"Shard {args.shard_id}: {ep_id} after Whisper | "
            f"english_kept={len(after_whisper)} non_english_dropped={len(after_music) - len(after_whisper)}"
        )
        if not after_whisper:
            stage_log(f"Shard {args.shard_id}: {ep_id} has no English segments after Whisper; marking done")
            mark_episode_done(ep_id)
            continue

        # ==== Stage 4: Batched Canary transcription ====
        stage_log(
            f"Shard {args.shard_id}: {ep_id} running Canary on {len(after_whisper)} segments"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_paths = []
            for i, (seg_id, seg, _, _) in enumerate(after_whisper):
                tmp_path = Path(tmp_dir) / f"{i}.wav"
                sf.write(str(tmp_path), seg.numpy(), SAMPLE_RATE)
                tmp_paths.append(str(tmp_path))

            canary_texts = batch_canary_transcribe(canary_model, tmp_paths)

        # ==== Stage 5: WER comparison and output ====
        filter_dict = {}
        for (seg_id, seg, info, wtext), ctext in zip(after_whisper, canary_texts):
            if not ctext:
                continue
            whisper_norm = re.sub(r'[^a-zA-Z0-9 ]', '', wtext).lower().strip()
            canary_norm = re.sub(r'[^a-zA-Z0-9 ]', '', ctext).lower().strip()

            if not whisper_norm or not canary_norm:
                total_wer_mismatch += 1
                continue

            
            wer_ab = wer(whisper_norm, canary_norm)
            wer_ba = wer(canary_norm, whisper_norm)
            agreement_error = max(wer_ab, wer_ba)
            
            if agreement_error > WER_THRESHOLD:
                total_wer_mismatch += 1
                continue
           

            duration = info["end"] - info["start"]
            filter_dict[seg_id] = {
                "start": info["start"],
                "end": info["end"],
                "duration": round(duration, 3),
                "snr": info["snr"],
                "c50": info["c50"],
                "music_sum": info["music_sum"],
                "whisper_large_v3_text": wtext,
                "canary_text": ctext,
                "agreement_wer": round(agreement_error, 4),
                # Preserve episode-level metadata
                "episode_index": ep_idx,
                "epTitle": ep_meta.get("epTitle", ""),
                "podTitle": ep_meta.get("podTitle", ""),
                "mp3url": ep_meta.get("mp3url", ""),
                "audio_file": Path(audio_path).name,
            }
            total_duration_kept += duration

        if not filter_dict:
            stage_log(f"Shard {args.shard_id}: {ep_id} kept 0 segments after WER; marking done")
            mark_episode_done(ep_id)
            continue

        output_file = Path(args.output_dir) / f"{ep_id}.json"
        with open(str(output_file), "w") as f:
            json.dump(filter_dict, f, indent=4)

        total_segments_kept += len(filter_dict)
        save_checkpoint()
        stage_log(
            f"Shard {args.shard_id}: completed {ep_id} | "
            f"kept_this_episode={len(filter_dict)} totals kept={total_segments_kept} "
            f"music={total_music_filtered} snr={total_snr_filtered} "
            f"hrs={total_duration_kept/3600:.2f}"
        )

    ep_pbar.close()

    # ---- Final summary ----
    print(f"\n{'='*60}")
    print(f"SPLIT [{args.start_idx}, {end_idx})  SHARD {args.shard_id}/{args.num_shards} COMPLETE")
    print(f"{'='*60}")
    print(f"Segments kept:              {total_segments_kept}")
    print(f"Speech duration kept:       {total_duration_kept/3600:.2f}h")
    print(f"Filtered by SNR (<{SNR_THRESHOLD}dB):   {total_snr_filtered}")
    print(f"Filtered by music (>{MUSIC_THRESHOLD}):  {total_music_filtered}")
    print(f"Filtered by WER mismatch:   {total_wer_mismatch}")
    print(f"Filtered by non-English:    {total_non_english}")
    print(f"Output: {args.output_dir}/")
