#!/bin/bash

curl -L "https://cloud.tsinghua.edu.cn/d/d812f7d1fc474b14bbd0/files/?p=%2FEEG-ImageNet_1.pth&dl=1" \
    -o ./data/EEG-ImageNet_1.pth

curl -L "https://cloud.tsinghua.edu.cn/d/d812f7d1fc474b14bbd0/files/?p=%2FEEG-ImageNet_2.pth&dl=1" -o \
    ./data/EEG-ImageNet_2.pth