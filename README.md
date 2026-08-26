# Podcast Transcript Verification Code

Anonymous code release for preprocessing NaturalVoices and SPoRC, generating
speaker and demographic verification trials, scoring those trials with
embedding and language models, and computing ROC-based metrics.

## Repository Layout

~~~text
.
├── data_processing/
│   ├── merge_label_shards.py
│   ├── naturalvoices/
│   │   ├── download_audio.sh
│   │   ├── filter_and_transcribe.py
│   │   ├── infer_accent.py
│   │   ├── build_speaker_info.py
│   │   └── data/speaker_info.json
│   └── sporc/
│       ├── extract_audio_urls.py
│       ├── download_audio.sh
│       ├── filter_and_transcribe.py
│       ├── infer_age_and_gender.py
│       ├── infer_accent.py
│       ├── build_speaker_info.py
│       └── data/speaker_info.json
├── experiments/
│   ├── common/
│   │   ├── score_trials.py
│   │   ├── evaluate_trials.py
│   │   └── check_scoring_progress.py
│   ├── naturalvoices/generate_trials.py
│   └── sporc/
│       ├── generate_topic_controlled_trials.py
│       └── generation_report.json
└── scripts/
    ├── generate_naturalvoices_trials.sh
    ├── generate_sporc_trials.sh
    ├── score_all.sh
    └── evaluate_all.sh
~~~

The two included speaker-info JSON files are the final preprocessing outputs
used as trial-generator inputs. Raw audio, raw dataset metadata, generated
trials, model scores, and plots are not duplicated in this repository.

## Installation

Python 3.10 or newer and a Linux-like environment are recommended.

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

Install the larger audio-processing stack only when rebuilding the cleaned
datasets:

~~~bash
pip install -r requirements-processing.txt
~~~

The preprocessing pipeline also requires FFmpeg on PATH and the Vox-Profile
package that provides WavLMWrapper and WhisperWrapper. GPU execution is
strongly recommended for ASR, audio classification, and neural embedding
models.

The exact demographic-checkpoint namespaces contain account identifiers and
are therefore not hardcoded in this anonymous release. Supply the same
checkpoints through `AGE_SEX_MODEL_ID` and `ACCENT_MODEL_ID` to reproduce
preprocessing. Trial generation, scoring, and evaluation do not require these
checkpoints because the final processed speaker metadata is included.

The included JSON files exceed ordinary Git hosting limits and are listed in
.gitattributes for Git LFS:

~~~bash
git lfs install
git lfs track "data_processing/*/data/*.json"
~~~

## Included Processed Data

- data_processing/naturalvoices/data/speaker_info.json: 2,344 speakers.
- data_processing/sporc/data/speaker_info.json: 10,165 speakers.

Each file is a JSON object keyed by speaker. An entry contains age, gender,
accent metadata, and one or more episode identifiers mapped to utterance
lists. These files contain no local filesystem paths. They retain public
dataset speaker and transcript content, so redistribution remains subject to
the original dataset licenses.

## Trial Protocols

### NaturalVoices

NaturalVoices can contain multiple speakers in one podcast. Its generator
therefore creates four conditions:

- easy positive (EP): same target label and same podcast;
- hard positive (HP): same target label and different podcasts;
- easy negative (EN): different target labels and different podcasts;
- hard negative (HN): different target labels and the same podcast.

For the speaker task, the target label is speaker identity. For gender, age,
and accent, every pair is cross-speaker and the target is the corresponding
attribute. Age is grouped into <30, 30--60, and 60+; accent is grouped into
North America and Other.

### SPoRC

The retained SPoRC episodes are single-speaker. The final protocol therefore
uses only HP and EN files. It pairs different episodes and controls topic with:

- TF-IDF cosine similarity over cleaned episode title and description,
  restricted to at most 0.10;
- Jaccard overlap over non-empty Apple Podcast categories;
- matched topic-similarity and category-overlap strata across classes.

Speaker negatives additionally match gender, age group, and accent group.
Speaker positives may use different episodes from the same RSS feed; all
cross-speaker pairs use different RSS feeds. Exact utterance overlap and
positive pairs that expose the speaker name on both sides are removed.
Positive and negative classes are balanced one-to-one.

experiments/sporc/generation_report.json records the exact protocol,
per-setting counts, retained/generated counts, RSS relations, and similarity
statistics for the paper trials.

## Generate Trials

The paper uses four input settings: 50_L5, 50_L20, 200_L5, and 200_L20. The
first number is the target number of probe words; L is the maximum number of
enrollment utterances.

Generate all NaturalVoices settings from the included speaker metadata:

~~~bash
bash scripts/generate_naturalvoices_trials.sh
~~~

An optional first argument changes the output root:

~~~bash
bash scripts/generate_naturalvoices_trials.sh outputs/trials/naturalvoices
~~~

Generate all topic-controlled SPoRC settings. The raw SPoRC JSONL is still
required because episode titles, descriptions, RSS URLs, and categories are
not duplicated in speaker_info.json:

~~~bash
bash scripts/generate_sporc_trials.sh \
  /path/to/sporc-single-speaker.jsonl \
  outputs/trials/sporc
~~~

The generators use seed 42. NaturalVoices samples at most 2,000 speakers.
SPoRC uses a 0.10 topic threshold, one speaker negative per positive, and
2,000 trials per class for each demographic task.

The paper trial construction optionally prioritized eligible pairs from an
earlier trial pool before filling the remaining strata. To repeat that
retention step, invoke the generator directly with `--existing_root` and, when
needed, `--additional_existing_roots`:

~~~bash
python experiments/sporc/generate_topic_controlled_trials.py \
  --raw_jsonl /path/to/sporc-single-speaker.jsonl \
  --existing_root /path/to/earlier/trials \
  --output_root outputs/trials/sporc
~~~

Without an earlier trial root, the wrapper generates a deterministic fresh
sample under the same filtering, matching, balancing, and random-seed
protocol. `experiments/sporc/generation_report.json` records the statistics of
the trial set used in the paper, including retained and newly sampled rows.

## Score Trials

score_trials.py supports five embedding methods and three LLM judges:

- sbert, tfidf, cisr, pcm, and luar;
- gpt, llama, and qwen.

The shell wrapper runs settings and tasks in order. Existing non-null scores
are skipped, so an interrupted run can be resumed with the same command.

Embedding example:

~~~bash
bash scripts/score_all.sh \
  sporc \
  outputs/trials/sporc \
  outputs/scores/sporc \
  sbert tfidf cisr pcm luar
~~~

For hosted models, export credentials in the shell. Never place keys in a
script or commit them to the repository.

~~~bash
export OPENAI_API_KEY="..."
export TOGETHER_API_KEY="..."
~~~

Run each hosted model separately when API limits are restrictive:

~~~bash
bash scripts/score_all.sh sporc \
  outputs/trials/sporc outputs/scores/sporc gpt

bash scripts/score_all.sh sporc \
  outputs/trials/sporc outputs/scores/sporc llama

bash scripts/score_all.sh sporc \
  outputs/trials/sporc outputs/scores/sporc qwen
~~~

The scorer requests a 0--100 confidence score by default. Together-hosted
models are executed sequentially, never concurrently. Set PAUSE_S only if an
API requires additional pacing:

~~~bash
PAUSE_S=1 bash scripts/score_all.sh sporc \
  outputs/trials/sporc outputs/scores/sporc llama
~~~

Use naturalvoices as the first argument and the NaturalVoices trial/score
roots for that dataset.

To inspect completion without making API calls:

~~~bash
python experiments/common/check_scoring_progress.py \
  --trial_root outputs/trials/sporc \
  --result_root outputs/scores/sporc \
  --splits hard_positives easy_negatives \
  --details
~~~

## Evaluate and Plot

Evaluate every setting and task:

~~~bash
bash scripts/evaluate_all.sh \
  outputs/trials/sporc \
  outputs/scores/sporc \
  outputs/evaluation/sporc
~~~

Each task directory receives ROC plots and summary.json. The summary reports
AUC, equal-error rate (EER), TPR at selected FPR values, and the epsilon-style
metric. NaturalVoices additionally receives valid easy/hard and
same-/different-podcast views. Views with only one class, including
nonexistent SPoRC conditions, are skipped.

## Rebuild NaturalVoices Speaker Metadata

Obtain all_data.json, audio_license.csv, and the audio archives from the
official NaturalVoices distribution. Download archives with:

~~~bash
bash data_processing/naturalvoices/download_audio.sh raw_audio_archives
~~~

After extraction, filter and transcribe eligible segments:

~~~bash
python data_processing/naturalvoices/filter_and_transcribe.py \
  --audio_dir /path/to/podcasts_flac \
  --metadata_json /path/to/all_data.json \
  --license_csv /path/to/audio_license.csv \
  --output_dir filtered_output/naturalvoices
~~~

Infer utterance-level accents. To keep this double-blind artifact anonymous,
the account-qualified checkpoint identifier is not hardcoded; supply the
checkpoint described in the submission:

~~~bash
export ACCENT_MODEL_ID="<anonymous-accent-checkpoint>"
python data_processing/naturalvoices/infer_accent.py \
  --filtered_dir filtered_output/naturalvoices \
  --audio_dir /path/to/podcasts_flac \
  --output_dir utterance_labels/naturalvoices
~~~

When accent inference is sharded, merge its outputs:

~~~bash
python data_processing/merge_label_shards.py \
  --input_glob "utterance_labels/naturalvoices/dialect.shard_*.json" \
  --output_file utterance_labels/naturalvoices/dialect.json
~~~

Build the trial-generator input:

~~~bash
python data_processing/naturalvoices/build_speaker_info.py \
  --metadata_json /path/to/all_data.json \
  --dialect_json utterance_labels/naturalvoices/dialect.json \
  --output_file data_processing/naturalvoices/data/speaker_info.json
~~~

## Rebuild SPoRC Speaker Metadata

Obtain sporc-single-speaker.jsonl from the official SPoRC distribution.
Extract and download the retained episode URLs:

~~~bash
python data_processing/sporc/extract_audio_urls.py \
  --metadata_jsonl /path/to/sporc-single-speaker.jsonl \
  --output_file raw_data/sporc/urls.txt

bash data_processing/sporc/download_audio.sh \
  raw_data/sporc/urls.txt raw_data/sporc/audio_files
~~~

Filter audio and run the dual-ASR agreement check:

~~~bash
python data_processing/sporc/filter_and_transcribe.py \
  --urls_file raw_data/sporc/urls.txt \
  --jsonl_file /path/to/sporc-single-speaker.jsonl \
  --audio_dir raw_data/sporc/audio_files \
  --output_dir filtered_output/sporc
~~~

Set the anonymized demographic checkpoint identifiers described in the
submission, then infer age/gender and accent:

~~~bash
export AGE_SEX_MODEL_ID="<anonymous-age-sex-checkpoint>"
export ACCENT_MODEL_ID="<anonymous-accent-checkpoint>"

python data_processing/sporc/infer_age_and_gender.py \
  --filtered_output_dir filtered_output/sporc \
  --audio_dir raw_data/sporc/audio_files \
  --output_dir utterance_labels/sporc

python data_processing/sporc/infer_accent.py \
  --filtered_output_dir filtered_output/sporc \
  --audio_dir raw_data/sporc/audio_files \
  --output_dir utterance_labels/sporc
~~~

Both inference programs accept --shard_id and --num_shards. The metadata
builder automatically merges dialect*.json and age_sex*.json shard files:

~~~bash
python data_processing/sporc/build_speaker_info.py \
  --jsonl_file /path/to/sporc-single-speaker.jsonl \
  --filtered_output_dir filtered_output/sporc \
  --utterance_labels_dir utterance_labels/sporc \
  --output_file data_processing/sporc/data/speaker_info.json
~~~

## Reproducibility Notes

- All commands use caller-provided or repository-relative paths.
- Code cleanup is limited to organization, descriptive names, command-line
  paths, credential handling, and removal of inactive duplicate code.
- The SPoRC filter stores only an audio filename in intermediate JSON rather
  than an author-machine path; downstream scripts resolve it with
  `--audio_dir`, leaving the final speaker metadata unchanged.
- API credentials are read only from environment variables.
- Model score JSON is updated atomically and can be resumed.
- Hosted LLMs run sequentially to avoid rate-limit amplification.
- Generated trials, scores, plots, caches, and raw audio are ignored by Git.
- The release contains no author names, machine usernames, cluster paths, or
  embedded API credentials.
