#!/bin/bash

echo "Running experiments with different seeds for inter protocol, all subjects"
for seed in 42 43 44
do
    echo "Running experiment with seed $seed"
    python train.py protocol=inter subject=-1 seed=$seed
done

echo "Running experiments with different seeds for intra protocol, all subjects"
for seed in 42 43 44
do
    echo "Running experiment with seed $seed"
    python train.py protocol=intra subject=-1 \
        eeg_t_start=0.0 eeg_t_end=0.7 n_timepoints=70 seed=$seed
done