#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${SOURCE_DIR:-data/things_eeg}"
DEST_DIR="${DEST_DIR:-data/things_eeg_63}"
if (( $# )); then
  SUBJECTS=("$@")
else
  SUBJECTS=(01 02 03 04 05 06 07 08 09 10)
fi

declare -A URLS=(
  [01]="https://osf.io/download/7gxvj/"
  [02]="https://osf.io/download/ycfq3/"
  [03]="https://osf.io/download/eqjbv/"
  [04]="https://osf.io/download/y3ghr/"
  [05]="https://osf.io/download/2we8r/"
  [06]="https://osf.io/download/3c7hm/"
  [07]="https://osf.io/download/k3xm7/"
  [08]="https://osf.io/download/hr3p5/"
  [09]="https://osf.io/download/jgh8q/"
  [10]="https://osf.io/download/d6pmy/"
)

if command -v aria2c >/dev/null 2>&1; then
  DOWNLOADER=aria2c
elif command -v curl >/dev/null 2>&1; then
  DOWNLOADER=curl
else
  echo "Either aria2c or curl is required." >&2
  exit 1
fi

mkdir -p "$DEST_DIR"

for name in image_metadata.npy training_images test_images image_feature; do
  if [[ ! -e "$DEST_DIR/$name" && ! -L "$DEST_DIR/$name" ]]; then
    if [[ ! -e "$SOURCE_DIR/$name" ]]; then
      echo "Missing shared asset: $SOURCE_DIR/$name" >&2
      exit 1
    fi
    ln -s "../$(basename "$SOURCE_DIR")/$name" "$DEST_DIR/$name"
  fi
done

for subject in "${SUBJECTS[@]}"; do
  subject="${subject#sub-}"
  subject="${subject#0}"
  printf -v subject "%02d" "$subject"

  if [[ -z "${URLS[$subject]:-}" ]]; then
    echo "Unknown subject: $subject" >&2
    exit 1
  fi

  archive="$DEST_DIR/sub-${subject}__63_channels.zip"
  echo "Downloading full-channel subject $subject..."
  if [[ "$DOWNLOADER" == "aria2c" ]]; then
    aria2c \
      --allow-overwrite=true \
      --auto-file-renaming=false \
      --continue=true \
      --dir "$DEST_DIR" \
      --out "$(basename "$archive")" \
      --summary-interval=30 \
      "${URLS[$subject]}"
  else
    curl --fail --location --continue-at - \
      --output "$archive" "${URLS[$subject]}"
  fi

  echo "Extracting $archive..."
  unzip -q -o "$archive" -d "$DEST_DIR"
  rm "$archive"

  extracted_dir="$DEST_DIR/sub-${subject}__63_channels"
  subject_dir="$DEST_DIR/sub-${subject}"
  if [[ -d "$extracted_dir" ]]; then
    rm -rf "$subject_dir"
    mv "$extracted_dir" "$subject_dir"
  fi
done

echo "Full-channel EEG is ready in $DEST_DIR."
