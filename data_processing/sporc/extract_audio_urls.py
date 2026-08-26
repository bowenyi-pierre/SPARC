# Output urls.txt that contains all mp3 files' URLs for downloading
import json

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Extract downloadable audio URLs from SPoRC metadata.")
parser.add_argument("--metadata_jsonl", type=Path, required=True)
parser.add_argument("--output_file", type=Path, required=True)
args = parser.parse_args()

data = []
with args.metadata_jsonl.open(encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line))


def clean_url(raw_url):
    # 1. Remove quotes and invisible whitespace
    url = raw_url.replace('"', '').strip()
    
    # 2. Fix the "tps://" or missing scheme issue
    if '://' in url:
        # Split the URL into two parts at the '://'
        scheme, rest_of_url = url.split('://', 1)
        
        # If the scheme isn't 'http' or 'https', force it to 'https'
        if scheme not in ['http', 'https']:
            url = f"https://{rest_of_url}"
    else:
        # If there is no '://' at all, safely add it
        url = f"https://{url}"
        
    return url


available_host_guest_urls = []

for pod in data:
    if pod['hostSpeakerLabels'] != "SPEAKER_DATA_UNAVAILABLE" or pod['guestSpeakerLabels'] != "SPEAKER_DATA_UNAVAILABLE":
        available_host_guest_urls.append(clean_url(pod['mp3url']))


# Write to urls.txt
args.output_file.parent.mkdir(parents=True, exist_ok=True)
with args.output_file.open('w', encoding='utf-8') as f:
    for url in available_host_guest_urls:
        f.write(f"{url}\n")

