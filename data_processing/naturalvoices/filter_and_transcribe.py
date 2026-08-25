"""
Dual-ASR pipeline with AST music filtering.
Optimized with batched inference and multi-GPU sharding.

All filesystem locations are supplied on the command line. Use --help for the
complete interface, including multi-GPU sharding options.
"""

import re
import json
import torch
import whisper
import tempfile
import argparse
import torchaudio
import numpy as np
import pandas as pd
import soundfile as sf

from jiwer import wer
from pathlib import Path
from tqdm import tqdm
from nemo.collections.speechlm2.models import SALM
from transformers import ASTFeatureExtractor, ASTForAudioClassification

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.WARNING, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

SAMPLE_RATE = 16000
MUSIC_THRESHOLD = 0.1
WER_THRESHOLD = 0.1
AST_BATCH_SIZE = 64
WHISPER_BATCH_SIZE = 64
CANARY_BATCH_SIZE = 64  # Canary is larger, use smaller batches

# ---- Music categories from AudioSet ontology ----
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


# =============================================================================
# Batched AST music detection
# =============================================================================
def batch_check_music(segments_np, ast_model, ast_extractor, music_idx_tensor, device, sr=16000):
    """
    Batch AST inference on multiple segments. Returns list of music_sum scores.
    For segments > 10s, uses the first 10s (fast approximation for batching).
    """
    target_len = 10 * sr
    prepared = []
    for audio in segments_np:
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        elif len(audio) > target_len:
            audio = audio[:target_len]
        prepared.append(audio)

    music_sums = []
    for i in range(0, len(prepared), AST_BATCH_SIZE):
        batch = prepared[i:i + AST_BATCH_SIZE]
        inputs = ast_extractor(batch, sampling_rate=sr, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = torch.sigmoid(ast_model(**inputs).logits)
        batch_music = probs[:, music_idx_tensor].sum(dim=1).cpu().tolist()
        music_sums.extend(batch_music)

    return music_sums


# =============================================================================
# Batched Whisper decode
# =============================================================================
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
            mel = whisper.log_mel_spectrogram(padded, n_mels=whisper_model.dims.n_mels)
            mels.append(mel)
        mel_batch = torch.stack(mels).to(device)
        options = whisper.DecodingOptions()
        decode_results = whisper.decode(whisper_model, mel_batch, options)
        for r in decode_results:
            results.append((r.language, r.text))
    return results


# =============================================================================
# Batched Canary transcription
# =============================================================================
def batch_canary_transcribe(canary_model, wav_paths, max_new_tokens=128):
    """
    Batch Canary SALM transcription.
    Returns list of transcribed text strings.
    """
    texts = []
    for i in range(0, len(wav_paths), CANARY_BATCH_SIZE):
        batch_paths = wav_paths[i:i + CANARY_BATCH_SIZE]
        prompts = [
            [{"role": "user", "content": f"Transcribe the following: {canary_model.audio_locator_tag}", "audio": [p]}]
            for p in batch_paths
        ]
        try:
            answer_ids = canary_model.generate(prompts=prompts, max_new_tokens=max_new_tokens)
            for ids in answer_ids:
                texts.append(canary_model.tokenizer.ids_to_text(ids.cpu()))
        except Exception:
            # Fallback: process one by one if batch fails
            for p in batch_paths:
                try:
                    ids = canary_model.generate(
                        prompts=[[{"role": "user", "content": f"Transcribe the following: {canary_model.audio_locator_tag}", "audio": [p]}]],
                        max_new_tokens=max_new_tokens,
                    )
                    texts.append(canary_model.tokenizer.ids_to_text(ids[0].cpu()))
                except Exception:
                    texts.append("")
    return texts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter NaturalVoices speech and transcribe it with two ASR models."
    )
    parser.add_argument("--audio_dir", type=Path, required=True,
                        help="Directory containing <episode_id>.flac files")
    parser.add_argument("--metadata_json", type=Path, required=True,
                        help="NaturalVoices all_data.json")
    parser.add_argument("--license_csv", type=Path, required=True,
                        help="NaturalVoices audio-license table")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory for filtered per-episode JSON files")
    parser.add_argument("--license_subset", default="cc-by",
                        help="License value to retain in addition to public domain")
    parser.add_argument("--music_threshold", type=float, default=MUSIC_THRESHOLD)
    parser.add_argument("--wer_threshold", type=float, default=WER_THRESHOLD)
    parser.add_argument("--whisper_model", default="turbo")
    parser.add_argument("--canary_model", default="nvidia/canary-qwen-2.5b")
    parser.add_argument(
        "--ast_model", default="MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    parser.add_argument("--shard_id", type=int, default=0, help="This worker's shard index (0-based)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of parallel workers")
    parser.add_argument("--ast_batch", type=int, default=AST_BATCH_SIZE)
    parser.add_argument("--whisper_batch", type=int, default=WHISPER_BATCH_SIZE)
    parser.add_argument("--canary_batch", type=int, default=CANARY_BATCH_SIZE)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard_id must be in [0, num_shards)")
    AST_BATCH_SIZE = args.ast_batch
    WHISPER_BATCH_SIZE = args.whisper_batch
    CANARY_BATCH_SIZE = args.canary_batch
    MUSIC_THRESHOLD = args.music_threshold
    WER_THRESHOLD = args.wer_threshold

    audio_license = pd.read_csv(args.license_csv, index_col=0)
    audio_license = audio_license[
        (audio_license[" license"] == args.license_subset)
        | (audio_license[" license"] == "public domain")
    ]

    with args.metadata_json.open(encoding="utf-8") as f:
        json_data = json.load(f)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_speech, total_duration = 0, 0
    total_music_filtered = 0
    total_wer_mismatch = 0
    total_non_english = 0

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        print('GPU available, use GPU')

    # ---- Load models ----
    print("Loading Whisper Turbo...")
    whisper_turbo_model = whisper.load_model(args.whisper_model).to(device)

    print("Loading Canary SALM...")
    canary_model = SALM.from_pretrained(args.canary_model)
    if device.type == "cuda":
        canary_model = canary_model.bfloat16()
    canary_model = canary_model.eval().to(device)

    print("Loading AST music detector...")
    ast_extractor = ASTFeatureExtractor.from_pretrained(args.ast_model)
    ast_model = ASTForAudioClassification.from_pretrained(args.ast_model).to(device)
    ast_model.eval()

    music_label_indices = []
    for idx, label in ast_model.config.id2label.items():
        if label in MUSIC_CATEGORIES:
            music_label_indices.append(int(idx))
    music_idx_tensor = torch.tensor(music_label_indices, device=device)
    print(f"All models loaded. Batch sizes: AST={AST_BATCH_SIZE}, Whisper={WHISPER_BATCH_SIZE}, Canary={CANARY_BATCH_SIZE}\n")

    # ---- Build file list and apply sharding ----
    files_to_process = sorted([
        f for f in json_data
        if f in audio_license.index.values
        and not (args.output_dir / f"{f}.json").exists()
        and (args.audio_dir / f"{f}.flac").exists()
    ])

    # Shard: each worker takes every num_shards-th file
    files_to_process = files_to_process[args.shard_id::args.num_shards]
    print(f"Shard {args.shard_id}/{args.num_shards}: {len(files_to_process)} files to process\n")

    # ---- Main loop ----
    file_pbar = tqdm(files_to_process, desc="Files", unit="file")
    for file in file_pbar:
        file_pbar.set_postfix_str(
            f"kept={total_speech} music={total_music_filtered} hrs={total_duration/3600:.2f}"
        )

        try:
            audio_data, sr = torchaudio.load(str((args.audio_dir / f"{file}.flac")))
        except Exception:
            continue

        if sr != SAMPLE_RATE:
            audio_data = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(audio_data)

        # ==== Stage 0: Collect candidate segments (basic metadata filters) ====
        candidates = []  # list of (utterance_id, seg_audio_tensor)
        for utterance_id, utt_info in json_data[file].items():
            if "speech_classification" not in utt_info:
                continue
            duration = utt_info["end"] - utt_info["start"]
            if not (utt_info["num_speakers"] == 1
                    and utt_info["speech_classification"] == "Speech"
                    and utt_info["SNR"] >= 15
                    and 3 < duration <= 15):
                continue
            start = int(utt_info["start"] * SAMPLE_RATE)
            end   = int(utt_info["end"] * SAMPLE_RATE)
            seg_audio = audio_data[0, start:end]
            candidates.append((utterance_id, seg_audio))

        if not candidates:
            continue

        # ==== Stage 1: Batched AST music filter ====
        seg_arrays = [seg.numpy() for _, seg in candidates]
        music_scores = batch_check_music(seg_arrays, ast_model, ast_extractor, music_idx_tensor, device)

        after_music = []
        for (utt_id, seg), mscore in zip(candidates, music_scores):
            if mscore <= MUSIC_THRESHOLD:
                after_music.append((utt_id, seg, mscore))
            else:
                total_music_filtered += 1

        if not after_music:
            continue

        # ==== Stage 2: Batched Whisper decode (language filter + transcription) ====
        whisper_segments = [seg for _, seg, _ in after_music]
        whisper_results = batch_whisper_decode(whisper_segments, whisper_turbo_model, device)

        after_whisper = []
        for (utt_id, seg, mscore), (lang, wtext) in zip(after_music, whisper_results):
            if lang == "en":
                after_whisper.append((utt_id, seg, mscore, wtext))
            else:
                total_non_english += 1

        if not after_whisper:
            continue

        # ==== Stage 3: Batched Canary transcription ====
        # Write all segments to temp files at once
        tmp_paths = []
        tmp_dir = tempfile.mkdtemp()
        for i, (utt_id, seg, _, _) in enumerate(after_whisper):
            tmp_path = Path(tmp_dir) / f"{i}.wav"
            sf.write(str(tmp_path), seg.numpy(), SAMPLE_RATE)
            tmp_paths.append(str(tmp_path))

        canary_texts = batch_canary_transcribe(canary_model, tmp_paths)

        # Clean up temp files
        for p in tmp_paths:
            Path(p).unlink(missing_ok=True)
        Path(tmp_dir).rmdir()

        # ==== Stage 4: WER comparison and output ====
        filter_dict = dict()
        for (utt_id, seg, mscore, wtext), ctext in zip(after_whisper, canary_texts):
            if not ctext:
                continue
            whisper_norm = re.sub(r'[^a-zA-Z0-9 ]', '', wtext).lower().strip()
            canary_norm  = re.sub(r'[^a-zA-Z0-9 ]', '', ctext).lower().strip()

            # In case whisper or canary only outputs punctuations
            if not whisper_norm or not canary_norm:
                total_wer_mismatch += 1
                continue

            # Make agreement symmetric
            wer_ab = wer(whisper_norm, canary_norm)
            wer_ba = wer(canary_norm, whisper_norm)
            agreement_error = max(wer_ab, wer_ba)
            
            if agreement_error > WER_THRESHOLD:
                total_wer_mismatch += 1
                continue

            duration = json_data[file][utt_id]["end"] - json_data[file][utt_id]["start"]
            filter_dict[utt_id] = dict(json_data[file][utt_id])
            filter_dict[utt_id]["whisper_turbo_text"] = wtext
            filter_dict[utt_id]["canary_text"] = ctext
            filter_dict[utt_id]["agreement_wer"] = agreement_error
            filter_dict[utt_id]["music_sum"] = mscore
            total_duration += duration

        if not filter_dict:
            continue

        with (args.output_dir / f"{file}.json").open("w", encoding="utf-8") as f:
            json.dump(filter_dict, f, indent=4)

        total_speech += len(filter_dict)

    file_pbar.close()

    print(f"\n{'='*60}")
    print(f"SHARD {args.shard_id}/{args.num_shards} COMPLETE")
    print(f"{'='*60}")
    print(f"Utterances kept:          {total_speech}")
    print(f"Speech duration:          {total_duration/3600:.2f}h")
    print(f"Filtered by music:        {total_music_filtered}")
    print(f"Filtered by WER mismatch: {total_wer_mismatch}")
    print(f"Filtered by non-English:  {total_non_english}")
    print(f"Output: {args.output_dir}/")
