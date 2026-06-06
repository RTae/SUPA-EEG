#!/bin/bash
set -e

BASE="https://cloud.tsinghua.edu.cn"
DIR="data/things_eeg"

if ! command -v aria2c >/dev/null 2>&1; then
  echo "aria2c is required but was not found in PATH."
  echo "Install it first, for example: sudo apt-get install aria2"
  exit 1
fi

mkdir -p $DIR

download() {
  local url="$BASE/$1/?dl=1"
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
  [01]="f/6e4851c36cd64656b051"
  [02]="f/51f23849a3cb40839148"
  [03]="f/e69ae8a471d04e7eb709"
  [04]="f/e7cfad28adc54e729b2c"
  [05]="f/9af915568c63485a9753"
  [06]="f/cf2c1819a012438fb829"
  [07]="f/0413846a1de64cc1bab2"
  [08]="f/7171774f2b324c9f889d"
  [09]="f/9afc6c00192c47528d01"
  [10]="f/0127c638be494f878e23"
)

for i in {01..10}; do
  zip_file="$DIR/sub-${i}_63.zip"
  sub_dir="$DIR/sub-${i}_63"

  if [[ -d "$sub_dir" ]]; then
    echo "Already extracted, skipping: $sub_dir"
    [[ -f "$zip_file" ]] && rm "$zip_file"
    continue
  fi

  download "${_URLS[$i]}" "$zip_file"

  echo "Extracting sub-${i}_63.zip → $sub_dir ..."
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