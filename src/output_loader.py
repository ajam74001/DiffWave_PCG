'''
This script loads the predictions of the Diffwave and do the segmentions around the S1 and S2 peaks 
the out put of this script is a csv file containing the segments of all the generated signals by DiffWave
'''
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import librosa
import librosa.display
import scipy.io
import scipy.io as sio
from torch.utils.data import Dataset
import torch
from filters import bandpass_filter # my files and functions 
import pywt
from utilities import calculate_rmssd, zero_crossing_ratio, ratio_of_normal_peaks
import soundfile as sf
import os
from segmentation import segment
# TODO :add the resampling stuff here in the data loader : DONE!
# TODO: add bandpass filter: DONE!
# TODO: add the quality assesment to the loader: DONE! 
class Data_loader():
    def __init__(self, path_sig, encode=True, resample=True):
        self.resample = resample
        self.path_sig = path_sig

    def len(self):
        return len(os.listdir(self.path_sig))
    
    def get_item(self, id):
        ret = {}
        # record_type = 'AV'
        # ret['subject_id'] = self.subject_ids[index]
        # ret['record_type'] = record_type
        path_root = f"{self.path_sig}/{id}"
        print(path_root)
#         audio , ret['sr'] = librosa.load(f"{path_root}.wav", mono=True, sr=None) # ret['sr'] is the sampling rate 
        try:
            audio , fs = librosa.load(f"{path_root}.wav", mono=True, sr=None)
            # print("fs", fs)
        except FileNotFoundError:
            # print(id)
            print('File not found!')
        #### Check the quality of the signal and label it- 0: good , 1: corrupted
        ## the quality criteras are adopted from paper "Analysis of PCG signals using quality assessment and homomorphic filters for localization and classification of heart sounds"
        # coeffs = pywt.wavedec(audio, 'db1', level=2)
        # cA2, cD2, cD1 = coeffs  # cA2 is the approximation coefficients at level 2
        # if calculate_rmssd(cA2)> 0.1 or zero_crossing_ratio(cA2)>0.3 or ratio_of_normal_peaks(cA2, fs )>0.5: # threshold is coming from a paper 
        #     ret['label']= 1 # signal is corrupted      
        # else:
        #     ret['label']= 0

        # filtering the audio by a band pass filter 
        audio = bandpass_filter(audio, fs)
        # standardization the audio data 
        audio = (audio - np.mean(audio))/ np.std(audio)
        
        if self.resample: 
            # audio = np.asfortranarray(audio)
            # D = np.abs(librosa.stft(audio))
            # max_bin_overall = np.argmax(np.sum(D, axis=1))
            # highest_frequency = librosa.fft_frequencies(sr=fs)[max_bin_overall]

            # Compute the spectrogram of the audio signal to the aim of resampling 
            audio = np.asfortranarray(audio)
            D = np.abs(librosa.stft(audio))
            # Find the frequency bin with the maximum energy
            max_bin = np.argmax(D, axis=0)

            # Convert the frequency bin to Hertz
            frequencies = librosa.fft_frequencies(sr=fs)
            highest_frequency = np.max(frequencies[max_bin]) # getting the highest frequency in the spectrum of the signal        ret['audio'] = librosa.resample(data['audio'], orig_sr=data['sr'], target_sr=highest_frequency*2)
            print("highest_frequency", highest_frequency)
            ret['audio'] = librosa.resample(audio, orig_sr= fs, target_sr=highest_frequency*2) # resampling procudure 
            ret['sr'] = highest_frequency*2
        else:
            ret['audio'] = audio
            ret['sr'] = fs
        # ret['murmur'], ret['outcome'] = self.demo_data[self.demo_data['Patient ID']==self.subject_ids[index]][['Murmur', 'Outcome']].iloc[0]
        return ret
        
# Run a Sample!
# path_demo = "../../Datasets/physionet.org/files/circor-heart-sound/1.0.3/training_data.csv"
path_sig = "/home/ainazj1/psanjay_ada/users/ainazj1/Datasets/physionet.org/files/circor-heart-sound/1.0.3/output_diffwave"
data_sample = Data_loader( path_sig,  resample= True) # set resample to False and repeat 
data = data_sample.get_item(100)

print(data_sample.len())
print(max(data['audio']), min(data['audio']))
print(data['sr'])
# all_data_seg = []
# for index in range(1,data_sample.len()+1):
#     data = data_sample.get_item(index)
#     seg = segment(data['audio'])
#     all_data_seg.append(seg)

# flattened_data = [item for sublist in all_data_seg for item in sublist]
# print(pd.DataFrame(flattened_data).shape)
# pd.DataFrame(flattened_data).to_csv('/home/ainazj1/psanjay_ada/users/ainazj1/Datasets/physionet.org/files/circor-heart-sound/1.0.3/fake_diffwave_final.csv')
