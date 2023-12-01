import numpy as np
from scipy.io import wavfile
import pandas as pd
from scipy.signal import find_peaks
from utilities import normalize_signal
import soundfile as sf
import tqdm
#TODO: adjust the SRs 
real = pd.read_csv("PCG_timeseris_normal_processed_NoResample.csv").drop('Unnamed: 0', axis=1) # changed to the last processed version 
labels = pd.read_csv('PCG_timeseries_labels.csv')


# Define the sample rate. For example, 44100 Hz is a common sample rate for audio.
sample_rate = 4000
# TODO: get rid of the nans and apply the segmentation algo, maybe u need to not doing any resampling in the data loader (not sure yet)
# getting the high quality signals 
indices_to_filter = labels.index[labels['0'] == 1].tolist() # filtering the corrupted signals 
real = real.drop(indices_to_filter)
# segmentation Algo normalizing 
final_data = [] 
win_len =1100
for indx in tqdm.tqdm(range(real.shape[0])):
    real1  = real.iloc[indx,:].dropna()
    real1 = real1.iloc[int(0.1*len(real1)):-int(0.1*len(real1)), ]
    real_norm = normalize_signal(real1)
    peaks, info = find_peaks(real_norm, height= [0.4,0.85]) # sounds good!
    
    segments=[]
    for i,p in enumerate(peaks):

            if real_norm[p-win_len:p+500]: # if it is non empty 
                    segments.append(real_norm[p-win_len:p+500])

    final_data.append(segments)

flattened_data = [item for sublist in final_data for item in sublist]
data = pd.DataFrame(flattened_data) # shape nx1600
print(data.shape)
for i in tqdm.tqdm(range(data.shape[0])):
    audio_data = data.iloc[i,:].values.astype(np.float32)
    # Name of the output .wav file
    output_file = f'../wave_processed/output_{i+1}.wav'
    # Write the data to a .wav file 
    sf.write(output_file, audio_data, sample_rate)
    # wavfile.write(output_file, sample_rate, audio_data)
