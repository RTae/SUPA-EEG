#!/bin/bash

for sub in {1..10}; do
  python scripts/preprocess_data.py --sub $sub --n_ses 4 --project_dir /workspace/time-series-project
done