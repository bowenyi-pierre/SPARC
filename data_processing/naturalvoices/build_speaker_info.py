"""
Build per-speaker metadata: transcripts, age, gender, and accent.

Accent comes from dialect.json (output of infer_dialect.py).
For each speaker, we average the dialect probability vectors across all their
utterances and take the argmax as the speaker-level accent label.
"""

import json
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path

# ---- Command line ----
parser = argparse.ArgumentParser(description="Build NaturalVoices speaker metadata.")
parser.add_argument("--metadata_json", type=Path, required=True,
                    help="NaturalVoices all_data.json")
parser.add_argument("--dialect_json", type=Path, required=True,
                    help="Merged utterance-level dialect predictions")
parser.add_argument("--output_file", type=Path, required=True)
args = parser.parse_args()

all_data_path = args.metadata_json
dialect_path = args.dialect_json
output_path = args.output_file

dialect_label_list = [
    'Eastern Asia', 'English', 'Germanic', 'Irish',
    'North America', 'Northern Irish', 'Oceania',
    'Other', 'Romance', 'Scottish', 'Semitic', 'Slavic',
    'South African', 'South-eastern Asia', 'Southern Asia', 'Welsh'
]

# ---- Load data ----
with open(all_data_path, 'r') as f:
    data = json.load(f)

with open(dialect_path, 'r') as f:
    dialect_data = json.load(f)

# ---- Build per-utterance dialect lookup: (episode, utterance_id) -> [label, probs] ----
# dialect_data structure: { file_id: { utterance_id: { dialect: { vox_profile: [label, probs] } } } }
utterance_dialect = {}
for file_id, utterances in dialect_data.items():
    for utterance_id, utt_info in utterances.items():
        if 'dialect' in utt_info and 'vox_profile' in utt_info['dialect']:
            utterance_dialect[(file_id, utterance_id)] = utt_info['dialect']['vox_profile']

print(f"Loaded {len(utterance_dialect)} utterance-level dialect predictions")

# ---- Build speaker_info ----
speaker_info = defaultdict(dict)
speaker_age_gender = defaultdict(lambda: defaultdict(list))
speaker_dialect_probs = defaultdict(list)  # speaker_id -> list of probability vectors

for epi, val in data.items():
    for turn, turn_info in val.items():

        speaker_id = None

        if turn_info.get('num_speakers') == 1:

            raw_speaker = turn_info.get("global_speaker")
            if raw_speaker is None:
                raw_speaker = turn_info.get("global_speaker2")

            if isinstance(raw_speaker, list):
                if len(raw_speaker) > 0:
                    speaker_id = raw_speaker[0]
            else:
                speaker_id = raw_speaker

            txt = turn_info.get("text")

            if speaker_id is not None and txt:

                # Episode transcripts
                if epi not in speaker_info[speaker_id]:
                    speaker_info[speaker_id][epi] = []
                speaker_info[speaker_id][epi].append(txt)

                # Age and gender stats
                age_gender = turn_info.get("age_gender", {})
                if age_gender:
                    speaker_age_gender[speaker_id]["age"].append(age_gender.get('Age', 0))
                    speaker_age_gender[speaker_id]["male"].append(age_gender.get('Male', 0))
                    speaker_age_gender[speaker_id]["female"].append(age_gender.get('Female', 0))
                    speaker_age_gender[speaker_id]["child"].append(age_gender.get('Child', 0)) # likely doesn't have any

                # Collect dialect probabilities for this utterance
                dialect_key = (epi, turn)
                if dialect_key in utterance_dialect:
                    _, probs = utterance_dialect[dialect_key]
                    # probs is a nested list [[p0, p1, ...]] from softmax — flatten if needed
                    if isinstance(probs[0], list):
                        probs = probs[0]
                    speaker_dialect_probs[speaker_id].append(probs)

# ---- Calculate age, gender, and accent per speaker ----
for speaker in list(speaker_info.keys()):

    # Age
    age_total = speaker_age_gender[speaker]['age']
    speaker_info[speaker]['age'] = sum(age_total) / len(age_total) if age_total else None

    # Gender
    gender_mean = {}
    female_scores = speaker_age_gender[speaker]['female']
    gender_mean['female'] = sum(female_scores) / len(female_scores) if female_scores else 0
    male_scores = speaker_age_gender[speaker]['male']
    gender_mean['male'] = sum(male_scores) / len(male_scores) if male_scores else 0
    child_scores = speaker_age_gender[speaker]['child']
    gender_mean['child'] = sum(child_scores) / len(child_scores) if child_scores else 0

    if any(gender_mean.values()):
        speaker_info[speaker]['gender'] = max(gender_mean, key=gender_mean.get)
    else:
        speaker_info[speaker]['gender'] = "Unknown"

    # Accent — average dialect probability vectors across all utterances, then argmax
    if speaker in speaker_dialect_probs and len(speaker_dialect_probs[speaker]) > 0:
        avg_probs = np.mean(speaker_dialect_probs[speaker], axis=0).tolist()
        accent_label = dialect_label_list[np.argmax(avg_probs)]
        speaker_info[speaker]['accent'] = accent_label
        speaker_info[speaker]['accent_probs'] = {
            label: round(p, 4) for label, p in zip(dialect_label_list, avg_probs)
        }
        speaker_info[speaker]['accent_num_utterances'] = len(speaker_dialect_probs[speaker])
    else:
        speaker_info[speaker]['accent'] = "Unknown"
        speaker_info[speaker]['accent_probs'] = {}
        speaker_info[speaker]['accent_num_utterances'] = 0

# ---- Summary stats ----
total_speakers = len(speaker_info)
speakers_with_accent = sum(1 for s in speaker_info.values() if s.get('accent') != "Unknown")
print(f"Total speakers: {total_speakers}")
print(f"Speakers with accent labels: {speakers_with_accent}")
print(f"Speakers without accent: {total_speakers - speakers_with_accent}")


# ---- Save ----
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as f:
    json.dump(dict(speaker_info), f, indent=2)

print(f"Saved to {output_path}")