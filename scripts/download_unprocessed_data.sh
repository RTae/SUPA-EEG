#!/bin/bash
set -e

BASE="https://files.osf.io/v1/resources/crxs4/providers/googledrive"
DIR="data/things_eeg"

if ! command -v aria2c >/dev/null 2>&1; then
  echo "aria2c is required but was not found in PATH."
  echo "Install it first, for example: sudo apt-get install aria2"
  exit 1
fi

mkdir -p $DIR

download() {
  local url="$BASE/$1"
  local out="$2"
  local out_dir
  local out_name
  out_dir="$(dirname "$out")"
  out_name="$(basename "$out")"
  if [[ -f "$out" ]]; then
    echo "Already exists, skipping: $out"
    return 0
  fi
  echo "Downloading $out..."
  aria2c \
    --allow-overwrite=true \
    --auto-file-renaming=false \
    --continue=true \
    --dir "$out_dir" \
    --out "$out_name" \
    --summary-interval=0 \
    "$url" || { echo "FAILED: $out"; exit 1; }
}

declare -A _URLS=(
  [01]="sub-01.zip"
  [02]="sub-02.zip"
  [03]="sub-03.zip"
  [04]="sub-04.zip"
  [05]="sub-05.zip"
  [06]="sub-06.zip"
  [07]="sub-07.zip"
  [08]="sub-08.zip"
  [09]="sub-09.zip"
  [10]="sub-10.zip"
)

for i in {01..10}; do
  zip_file="$DIR/sub-${i}_unprocessed.zip"
  sub_dir="$DIR/sub-${i}_unprocessed"

  if [[ -d "$sub_dir" ]]; then
    echo "Already extracted, skipping: $sub_dir"
    [[ -f "$zip_file" ]] && rm "$zip_file"
    continue
  fi

  download "${_URLS[$i]}" "$zip_file"

  echo "Extracting sub-${i}_unprocessed.zip → $sub_dir ..."
  tmp_dir="$DIR/.tmp_extract_${i}"
  mkdir -p "$tmp_dir"
  unzip -q "$zip_file" -d "$tmp_dir"
  # rename whatever the zip called its top-level folder to the desired name
  extracted=$(find "$tmp_dir" -maxdepth 1 -mindepth 1 -type d | head -1)
  mv "$extracted" "$sub_dir"
  rm -rf "$tmp_dir"
  rm "$zip_file"
done

echo "Done."