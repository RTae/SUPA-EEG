#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
MAX_SECONDS="${MAX_SECONDS:-6600}"
EPOCHS="${EPOCHS:-15}"
SUBJECT="${SUBJECT:-1}"
PROTOCOL="${PROTOCOL:-intra}"
OUT_ROOT="${OUT_ROOT:-outputs/bounded_ablations/$(date +%Y%m%d_%H%M%S)}"
FEATURE_ROOT="${FEATURE_ROOT:-data/things_eeg/image_feature}"
mkdir -p "$OUT_ROOT/logs"

run_one() {
  local name="$1"
  shift
  printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$name $*" >> "$OUT_ROOT/manifest.tsv"
  CUDA_VISIBLE_DEVICES="$GPU" timeout --signal=TERM --kill-after=60 "$MAX_SECONDS" \
    "$PYTHON" train.py \
      protocol="$PROTOCOL" subject="$SUBJECT" epochs="$EPOCHS" stage1_epochs="$EPOCHS" \
      eval_every=1 early_stop_patience=2 batch_size=1024 \
      skip_feature_extraction=true \
      "hydra.run.dir=$OUT_ROOT/$name" "$@" \
      > "$OUT_ROOT/logs/$name.log" 2>&1
}

case "${1:-all}" in
  exp2)
    # Balanced fractional design: all EEG encoders on InternViT, then the
    # baseline and best compact encoder on each additional available backbone.
    for eeg in eegproject eegnet tsconv eegconformer atm; do
      run_one "exp2_${eeg}_internvit" eeg_encoder_type="$eeg"
    done
    for bank in "$FEATURE_ROOT"/*/internvit_features.npy "$FEATURE_ROOT"/*/*.pt; do
      [[ -f "$bank" ]] || continue
      [[ "$bank" == *internvit_multilevel_20_24_28_32_36* ]] && continue
      tag="$(basename "$(dirname "$bank")" | tr -cs '[:alnum:]' '_')"
      dim="$("$PYTHON" scripts/inspect_feature_bank.py "$bank" --field dim)"
      layers="$("$PYTHON" scripts/inspect_feature_bank.py "$bank" --field layers)"
      for eeg in eegproject tsconv; do
        run_one "exp2_${eeg}_${tag}" eeg_encoder_type="$eeg" \
          image_feature_path="$bank" image_input_dim="$dim" n_layers="$layers" \
          "layer_ids=[$(seq -s, 0 $((layers - 1)))]"
      done
    done
    ;;
  exp3)
    run_one exp3_router image_layer_mode=router
    run_one exp3_uniform image_layer_mode=uniform
    run_one exp3_layer20 image_layer_mode=single image_layer_index=0
    run_one exp3_layer28 image_layer_mode=single image_layer_index=2
    run_one exp3_layer36 image_layer_mode=single image_layer_index=4
    ;;
  exp6)
    run_one exp6_1000hz eeg_suffix=_1khz_17 n_timepoints=1000 \
      temporal_compression=100 smooth_kernel_size=51 smooth_sigma=10.0
    ;;
  all)
    "$0" exp2
    "$0" exp3
    "$0" exp6
    ;;
  *)
    echo "usage: $0 {exp2|exp3|exp6|all}" >&2
    exit 2
    ;;
esac
