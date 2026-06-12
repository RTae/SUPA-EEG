#!/bin/bash

# Shared Encoder Ablation — Experiment 5
# Compares share_encoder_type: linear | none | separate | transformer | jepa

echo "=== Shared Encoder Ablation: inter protocol, all subjects ==="
for variant in linear none separate transformer jepa
do
    echo "--- share_encoder_type=$variant ---"
    python train.py protocol=inter subject=-1 share_encoder_type=$variant
done

echo "=== Shared Encoder Ablation: intra protocol, all subjects ==="
for variant in linear none separate transformer jepa
do
    echo "--- share_encoder_type=$variant ---"
    python train.py protocol=intra subject=-1 \
        eeg_t_start=0.0 eeg_t_end=0.7 n_timepoints=70 \
        share_encoder_type=$variant
done
