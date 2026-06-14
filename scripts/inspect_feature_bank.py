#!/usr/bin/env python
import argparse

import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("--field", choices=("dim", "layers"), required=True)
args = parser.parse_args()

if args.path.endswith(".pt"):
    payload = torch.load(args.path, map_location="cpu", weights_only=True)
    features = payload.get("features", payload)
    value = next(iter(features.values()))
    sample = np.stack([np.asarray(value[key]) for key in ("S1", "S2", "S3")])
else:
    features = np.load(args.path, allow_pickle=True).item()
    sample = np.asarray(next(iter(features.values())))
if sample.ndim == 1:
    sample = sample[None, :]
print(sample.shape[-1] if args.field == "dim" else sample.shape[0])
