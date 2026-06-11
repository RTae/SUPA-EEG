#!/bin/bash

echo "Running experiments with higher sampling rate (1000Hz) for inter protocol, all subjects"
for channel in 17 63
do
    echo "Running experiment with channel $channel"
    python train.py protocol=inter subject=-1 eeg_suffix=_1khz_${channel} n_channels=$channel
done

echo "Running experiments with higher sampling rate (1000Hz) for intra protocol, all subjects"
for channel in 17 63
do
    echo "Running experiment with channel $channel"
    python train.py protocol=intra subject=-1 eeg_suffix=_1khz_${channel} n_channels=$channel \
        eeg_t_start=0.0 eeg_t_end=0.7 n_timepoints=70
done