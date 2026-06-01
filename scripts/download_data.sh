# Check and install aria2c if not present
if ! command -v aria2c &> /dev/null; then
  echo "aria2c not found, installing..."
  apt-get install -y aria2 2>/dev/null || yum install -y aria2 2>/dev/null || pip install aria2p -q
fi

mkdir -p data/things_eeg
mkdir -p data/vision_encoder/clip

# Write input file for aria2c
# EEG Data
cat > /tmp/things_eeg.txt << 'EOF'
https://cloud.tsinghua.edu.cn/f/3f9f369660834eb49a4d/?dl=1
  out=sub-01.zip
https://cloud.tsinghua.edu.cn/f/7ed84ca62fa54b439e18/?dl=1
  out=sub-02.zip
https://cloud.tsinghua.edu.cn/f/f880d1eb0f964ad99c98/?dl=1
  out=sub-03.zip
https://cloud.tsinghua.edu.cn/f/51bf91e55c5f4efb8609/?dl=1
  out=sub-04.zip
https://cloud.tsinghua.edu.cn/f/85098648b4604d55968f/?dl=1
  out=sub-05.zip
https://cloud.tsinghua.edu.cn/f/092caa007a9845d9bc38/?dl=1
  out=sub-06.zip
https://cloud.tsinghua.edu.cn/f/9f052176ac0f4f25a885/?dl=1
  out=sub-07.zip
https://cloud.tsinghua.edu.cn/f/4c9ff435f1904e209bed/?dl=1
  out=sub-08.zip
https://cloud.tsinghua.edu.cn/f/70bea1e5fdb4401e930f/?dl=1
  out=sub-09.zip
https://cloud.tsinghua.edu.cn/f/ea778895483749f488d1/?dl=1
  out=sub-10.zip
https://cloud.tsinghua.edu.cn/f/c67e4ace9fbd46618717/?dl=1
  out=train_images.zip
https://cloud.tsinghua.edu.cn/f/4b56fa976f5e4a70b249/?dl=1
  out=test_images.zip
https://cloud.tsinghua.edu.cn/f/bb5a66919a524bb6832d/?dl=1
  out=image_metadata.npy
EOF

aria2c --dir=data/things_eeg \
  --input-file=/tmp/things_eeg.txt \
  --max-concurrent-downloads=4 --split=4 --min-split-size=10M

# Vision Encoder Data using a openai/clip-vit-base-patch32
aria2c --dir=data/things_eeg/image_feature/clip \
  --max-concurrent-downloads=4 --split=4 --min-split-size=10M \
  "https://cloud.tsinghua.edu.cn/f/7c0d0012439b49c5a512/?dl=1" -o visual_features_clip.pt

# Vision Encoder Data using a OpenGVLab/InternViT-6B-448px-V1-5
aria2c --dir=data/things_eeg/image_feature/internvit_multilevel_20_24_28_32_36 \
  --max-concurrent-downloads=4 --split=4 --min-split-size=10M \
  "https://cloud.tsinghua.edu.cn/f/bde721733abe4b1a9d4e/?dl=1" -o visual_features_internvit.pt

for i in {01..10}; do
  unzip data/things_eeg/sub-$i.zip -d data/things_eeg/
done
unzip data/things_eeg/train_images.zip -d data/things_eeg/
unzip data/things_eeg/test_images.zip -d data/things_eeg/

rm data/things_eeg/*.zip