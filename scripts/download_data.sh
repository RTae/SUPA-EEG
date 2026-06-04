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
mkdir -p $DIR/image_feature/clip
mkdir -p $DIR/image_feature/internvit_multilevel_20_24_28_32_36

download() {
  local url="$BASE/$1/?dl=1"
  local out="$2"
  local out_dir
  local out_name
  out_dir="$(dirname "$out")"
  out_name="$(basename "$out")"
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

# EEG subjects
download "f/3f9f369660834eb49a4d" "$DIR/sub-01.zip"
download "f/7ed84ca62fa54b439e18" "$DIR/sub-02.zip"
download "f/f880d1eb0f964ad99c98" "$DIR/sub-03.zip"
download "f/51bf91e55c5f4efb8609" "$DIR/sub-04.zip"
download "f/85098648b4604d55968f" "$DIR/sub-05.zip"
download "f/092caa007a9845d9bc38" "$DIR/sub-06.zip"
download "f/9f052176ac0f4f25a885" "$DIR/sub-07.zip"
download "f/4c9ff435f1904e209bed" "$DIR/sub-08.zip"
download "f/70bea1e5fdb4401e930f" "$DIR/sub-09.zip"
download "f/ea778895483749f488d1" "$DIR/sub-10.zip"

# Images + metadata
download "f/c67e4ace9fbd46618717" "$DIR/train_images.zip"
download "f/4b56fa976f5e4a70b249" "$DIR/test_images.zip"
download "f/bb5a66919a524bb6832d" "$DIR/image_metadata.npy"

# Vision features
download "f/7c0d0012439b49c5a512" "$DIR/image_feature/clip/visual_features_clip.pt"
download "f/bde721733abe4b1a9d4e" "$DIR/image_feature/internvit_multilevel_20_24_28_32_36/visual_features_internvit.npy"

# Extract
for i in {01..10}; do
  echo "Extracting sub-$i.zip..."
  unzip -q "$DIR/sub-$i.zip" -d "$DIR" && rm "$DIR/sub-$i.zip"
done

echo "Extracting images..."
unzip -q "$DIR/train_images.zip" -d "$DIR" && rm "$DIR/train_images.zip"
unzip -q "$DIR/test_images.zip"  -d "$DIR" && rm "$DIR/test_images.zip"

echo "Done."