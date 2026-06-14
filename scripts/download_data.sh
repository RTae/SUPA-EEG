#!/bin/bash
set -e

BASE="https://cloud.tsinghua.edu.cn"
DIR="data/things_eeg"

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
  if [[ -f "$out" ]]; then
    echo "Already exists, skipping: $out"
    return 0
  fi
  echo "Downloading $out..."
  if command -v aria2c >/dev/null 2>&1; then
    aria2c \
      --allow-overwrite=true \
      --auto-file-renaming=false \
      --continue=true \
      --dir "$out_dir" \
      --out "$out_name" \
      --summary-interval=0 \
      "$url"
  else
    curl --fail --location --retry 5 --retry-delay 5 \
      --continue-at - --output "$out" "$url"
  fi || { echo "FAILED: $out"; exit 1; }
}

declare -A EEG_URLS=(
  [01]="f/3f9f369660834eb49a4d" [02]="f/7ed84ca62fa54b439e18"
  [03]="f/f880d1eb0f964ad99c98" [04]="f/51bf91e55c5f4efb8609"
  [05]="f/85098648b4604d55968f" [06]="f/092caa007a9845d9bc38"
  [07]="f/9f052176ac0f4f25a885" [08]="f/4c9ff435f1904e209bed"
  [09]="f/70bea1e5fdb4401e930f" [10]="f/ea778895483749f488d1"
)
SUBJECTS="${SUBJECTS:-01 02 03 04 05 06 07 08 09 10}"
for i in $SUBJECTS; do
  download "${EEG_URLS[$i]}" "$DIR/sub-${i}.zip"
done

# Images + metadata
if [[ "${SKIP_IMAGES:-0}" != "1" ]]; then
  download "f/c67e4ace9fbd46618717" "$DIR/train_images.zip"
  download "f/4b56fa976f5e4a70b249" "$DIR/test_images.zip"
fi
download "f/bb5a66919a524bb6832d" "$DIR/image_metadata.npy"

# Vision features
download "f/7c0d0012439b49c5a512" "$DIR/image_feature/clip/visual_features_clip.pt"
download "f/bde721733abe4b1a9d4e" "$DIR/image_feature/internvit_multilevel_20_24_28_32_36/internvit_features.npy"

# Extract
for i in $SUBJECTS; do
  zip_file="$DIR/sub-${i}.zip"
  sub_dir="$DIR/sub-${i}"
  if [[ -d "$sub_dir" ]]; then
    echo "Already extracted, skipping: $sub_dir"
    [[ -f "$zip_file" ]] && rm "$zip_file"
    continue
  fi
  if [[ ! -f "$zip_file" ]]; then
    echo "WARNING: $zip_file not found, skipping."
    continue
  fi
  echo "Extracting sub-${i}.zip..."
  unzip -q "$zip_file" -d "$DIR" && rm "$zip_file"
done

if [[ "${SKIP_IMAGES:-0}" == "1" ]]; then
  echo "Skipping image archives (pre-extracted features are sufficient for training)."
elif [[ -d "$DIR/training_images" ]]; then
  echo "Already extracted, skipping: $DIR/training_images"
  [[ -f "$DIR/train_images.zip" ]] && rm "$DIR/train_images.zip"
else
  unzip -q "$DIR/train_images.zip" -d "$DIR" && rm "$DIR/train_images.zip"
fi

if [[ "${SKIP_IMAGES:-0}" == "1" ]]; then
  :
elif [[ -d "$DIR/test_images" ]]; then
  echo "Already extracted, skipping: $DIR/test_images"
  [[ -f "$DIR/test_images.zip" ]] && rm "$DIR/test_images.zip"
else
  unzip -q "$DIR/test_images.zip" -d "$DIR" && rm "$DIR/test_images.zip"
fi

echo "Done."
