
mkdir -p data/things_eeg
# Download the dataset from the given URL with loop
wget https://cloud.tsinghua.edu.cn/f/3f9f369660834eb49a4d/?dl=1 -O data/things_eeg/sub-01.zip
wget https://cloud.tsinghua.edu.cn/f/7ed84ca62fa54b439e18/?dl=1 -O data/things_eeg/sub-02.zip
wget https://cloud.tsinghua.edu.cn/f/f880d1eb0f964ad99c98/?dl=1 -O data/things_eeg/sub-03.zip
wget https://cloud.tsinghua.edu.cn/f/51bf91e55c5f4efb8609/?dl=1 -O data/things_eeg/sub-04.zip
wget https://cloud.tsinghua.edu.cn/f/171a344be8fb4f14a6e9/?dl=1 -O data/things_eeg/sub-05.zip
wget https://cloud.tsinghua.edu.cn/f/092caa007a9845d9bc38/?dl=1 -O data/things_eeg/sub-06.zip
wget https://cloud.tsinghua.edu.cn/f/9f052176ac0f4f25a885/?dl=1 -O data/things_eeg/sub-07.zip
wget https://cloud.tsinghua.edu.cn/f/4c9ff435f1904e209bed/?dl=1 -O data/things_eeg/sub-08.zip
wget https://cloud.tsinghua.edu.cn/f/70bea1e5fdb4401e930f/?dl=1 -O data/things_eeg/sub-09.zip
wget https://cloud.tsinghua.edu.cn/f/ea778895483749f488d1/?dl=1 -O data/things_eeg/sub-10.zip
wget https://cloud.tsinghua.edu.cn/f/c67e4ace9fbd46618717/?dl=1 -O data/things_eeg/train_images.zip
wget https://cloud.tsinghua.edu.cn/f/4b56fa976f5e4a70b249/?dl=1 -O data/things_eeg/test_images.zip
wget https://cloud.tsinghua.edu.cn/f/153e36193f9f473cb449/?dl=1 -O data/things_eeg/image_metadata.npy

# Unzip the downloaded files with loop
for i in {01..10}
do
  unzip data/things_eeg/sub-$i.zip -d data/things_eeg/
done
unzip data/things_eeg/train_images.zip -d data/things_eeg/
unzip data/things_eeg/test_images.zip -d data/things_eeg/

# Remove the zip files after unzipping
rm data/things_eeg/*.zip