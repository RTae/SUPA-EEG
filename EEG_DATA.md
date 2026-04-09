# What is EEG data

 Each EEG data sample has a size of (nchannels, fs · T ), where 
 - nchannels is the number of EEG electrodes, which is 62 in our dataset
   - The reference electrode name is in `data/mode/montage_ch_names.json`
   - Also the corresponding 3D coordinates of the electrodes are in `data/mode/montage.fif`
 - fs is the sampling frequency of the device, which is 1000 Hz in our dataset
 - T is the time window size, which in our dataset is the duration of the image stimulus presentation, i.e., 500 ms