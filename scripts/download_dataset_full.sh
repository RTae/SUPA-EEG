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

download "f/6e4851c36cd64656b051" "$DIR/sub-01_63.zip"
download "f/51f23849a3cb40839148" "$DIR/sub-02_63.zip"
download "f/26e1fd5a8eb440c8bf3b" "$DIR/sub-03_63.zip"
download "f/e7cfad28adc54e729b2c" "$DIR/sub-04_63.zip"
download "f/9af915568c63485a9753" "$DIR/sub-05_63.zip"
download "f/cf2c1819a012438fb829" "$DIR/sub-06_63.zip"
download "f/0413846a1de64cc1bab2" "$DIR/sub-07_63.zip"
download "f/7171774f2b324c9f889d" "$DIR/sub-08_63.zip"
download "f/9afc6c00192c47528d01" "$DIR/sub-09_63.zip"
download "f/0127c638be494f878e23" "$DIR/sub-10_63.zip"

# Extract
for i in {01..10}; do
  zip_file="$DIR/sub-${i}_63.zip"
  sub_dir="$DIR/sub-${i}_63"
  if [[ -d "$sub_dir" ]]; then
    echo "Already extracted, skipping: $sub_dir"
    [[ -f "$zip_file" ]] && rm "$zip_file"
    continue
  fi
  if [[ ! -f "$zip_file" ]]; then
    echo "WARNING: $zip_file not found, skipping."
    continue
  fi
  echo "Extracting sub-${i}_63.zip..."
  unzip -q "$zip_file" -d "$DIR" && rm "$zip_file"
done

echo "Done."