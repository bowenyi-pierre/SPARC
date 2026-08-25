import json
import yaml
import torch
import torchaudio
import numpy as np
import argparse, logging
import torch.nn.functional as F
import copy, time, pickle, shutil, sys, os, pdb

from tqdm import tqdm
from pathlib import Path

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model', 'dialect'))
torchaudio.set_audio_backend("soundfile")
# from whisper_dialect import WhisperWrapper

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

SAMPLE_RATE = 16000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer NaturalVoices accent labels.")
    parser.add_argument("--filtered_dir", type=Path, required=True)
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_id", default=os.environ.get("ACCENT_MODEL_ID"),
                        help="Vox-profile accent checkpoint (or set ACCENT_MODEL_ID)")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard_id must be in [0, num_shards)")
    if not args.model_id:
        parser.error("--model_id or ACCENT_MODEL_ID is required")

    split_path = args.filtered_dir
    filter_audio_path = args.audio_dir
    synthetic_labels = args.output_dir
    synthetic_labels.mkdir(parents=True, exist_ok=True)

    # Find device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available(): print('GPU available, use GPU')
    
    best_dict, acc_labels = dict(), dict()
    dialect_label_list = [
        'Eastern Asia', 'English', 'Germanic', 'Irish', 
        'North America', 'Northern Irish', 'Oceania', 
        'Other', 'Romance', 'Scottish', 'Semitic', 'Slavic', 
        'South African', 'South-eastern Asia', 'Southern Asia', 'Welsh'
    ]
    from vox_profile.src.model.accent.whisper_accent import WhisperWrapper

    whisper_model = WhisperWrapper.from_pretrained(
        args.model_id
    ).to(device)
    whisper_model.eval()
    
    split_files = sorted(os.listdir(split_path))[args.shard_id::args.num_shards]

    # Each shard writes independently, avoiding cross-process write races.
    if args.num_shards == 1:
        dialect_json_path = synthetic_labels / "dialect.json"
    else:
        dialect_json_path = (
            synthetic_labels
            / f"dialect.shard_{args.shard_id}_of_{args.num_shards}.json"
        )
    if dialect_json_path.exists():
        with open(dialect_json_path, "r") as f:
            dialect_dict = json.load(f)
        print(f"Resuming: {len(dialect_dict)} episodes already processed")
    else:
        dialect_dict = dict()

    for file in split_files:
        if ".json" not in file: continue
        file_id = file.replace('.json', '')

        # Skip already processed episodes
        if file_id in dialect_dict:
            continue

        with open(str(Path(split_path).joinpath(file)), "r") as f:
            json_data = json.load(f)

        # Load the full FLAC once per episode
        flac_path = Path(filter_audio_path).joinpath(f"{file_id}.flac")
        if not flac_path.exists():
            logging.warning(f"FLAC not found: {flac_path}, skipping")
            continue

        try:
            full_audio, sr = torchaudio.load(str(flac_path))
        except Exception as e:
            logging.warning(f"Error loading {flac_path}: {e}, skipping")
            continue

        # Resample to 16kHz if needed
        if sr != SAMPLE_RATE:
            full_audio = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(full_audio)

        dialect_dict[file_id] = dict()

        for utterance_id in tqdm(json_data, desc=file_id):
            # Slice the utterance from the full audio using start/end timestamps
            start_sample = int(json_data[utterance_id]["start"] * SAMPLE_RATE)
            end_sample   = int(json_data[utterance_id]["end"] * SAMPLE_RATE)
            audio_data   = full_audio[:, start_sample:end_sample].to(device)

            dialect_dict[file_id][utterance_id] = dict()
            dialect_dict[file_id][utterance_id]['dialect'] = dict()
            
            with torch.no_grad():
                whisper_logits, whisper_embeddings = whisper_model(audio_data, return_feature=True)
                ensemble_logits = whisper_logits
                ensemble_prob   = F.softmax(ensemble_logits, dim=1).detach().cpu().numpy().astype(float).tolist()

            dialect_label = dialect_label_list[np.argmax(ensemble_prob)]
            dialect_dict[file_id][utterance_id]['dialect']['vox_profile'] = [dialect_label, ensemble_prob]
        
        # Save after each episode
        with dialect_json_path.open("w", encoding="utf-8") as handle:
            json.dump(dialect_dict, handle, indent=4)

    logging.info(f"Done! Dialect results saved to {Path(synthetic_labels) / 'dialect.json'}")