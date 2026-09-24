#!/usr/bin/env bash
# Download every parquet asset from nflverse-data GitHub releases into data/raw/<release>/.
# Re-run anytime; curl -z skips files that haven't changed.
set -euo pipefail
cd "$(dirname "$0")/.."
curl -s "https://api.github.com/repos/nflverse/nflverse-data/releases?per_page=100" \
| python3 -c '
import json,sys
for r in json.load(sys.stdin):
    for a in r["assets"]:
        if a["name"].endswith(".parquet"):
            print(r["tag_name"], a["name"], a["browser_download_url"])
' | while read -r tag name url; do
  mkdir -p "data/raw/$tag"
  echo "data/raw/$tag/$name $url"
done | xargs -P 12 -n 2 sh -c 'curl -sSL --retry 3 -o "$0" "$1" && echo "ok $0"' | wc -l
