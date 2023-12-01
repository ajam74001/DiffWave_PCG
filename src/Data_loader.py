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

# TODO :add the resampling stuff here in the data loader : DONE!
# TODO: add bandpass filter: DONE!
# TODO: add the quality assesment to the loader: DONE! 
class Data_loader():
    def __init__(self, path_demo, path_sig, encode=True, resample=True):
        self.resample = resample
        self.path_sig = path_sig
        self.demo_data = pd.read_csv(path_demo) # example:~/courses/bio-sig/datasets/D1/physionet.org/files/circor-heart-sound/1.0.3/training_data.csv
        self.subject_ids = self.demo_data[self.demo_data['Outcome'] == "Normal"]['Patient ID'] 
        # getting all the subjects that all the recordings are available # check if only AV-PV-TV-MV has enough of data to proceed wiht 
        self.subject_ids = self.subject_ids.reset_index(drop=True)
        self.demo_data = self.demo_data[["Patient ID", "Murmur", "Outcome"]]
        if encode:
            label_encoder= LabelEncoder()
            self.demo_data['Murmur']= label_encoder.fit_transform(self.demo_data['Murmur']) # Murmur is absent/present/Unknown 
            self.demo_data['Outcome']= label_encoder.fit_transform(self.demo_data['Outcome']) # Diagnosed by the medical expert: normal/abnormal


    def len(self):
        return self. subject_ids.shape[0]
    
    def get_item(self, index ):
        ret = {}
        record_type = 'AV'
        ret['subject_id'] = self.subject_ids[index]
        ret['record_type'] = record_type
        path_root = f"{self.path_sig}/{self.subject_ids[index]}_{record_type}"
#         audio , ret['sr'] = librosa.load(f"{path_root}.wav", mono=True, sr=None) # ret['sr'] is the sampling rate 
        try:
            audio , fs = librosa.load(f"{path_root}.wav", mono=True, sr=None)
            # print("fs", fs)
        except FileNotFoundError:
            try:
                path_root = f"{self.path_sig}/{self.subject_ids[index]}_{'PV'}"
                audio , fs = librosa.load(f"{path_root}.wav", mono=True, sr=None)
                # print("fs", fs)

            except FileNotFoundError:
                try:
                    path_root = f"{self.path_sig}/{self.subject_ids[index]}_{'TV'}"
                    audio , fs = librosa.load(f"{path_root}.wav", mono=True, sr=None)
                    # print("fs", fs)
                except FileNotFoundError:
                    try:
                        path_root = f"{self.path_sig}/{self.subject_ids[index]}_{'MV'}"
                        audio , fs = librosa.load(f"{path_root}.wav", mono=True, sr=None)
                        # print("fs", fs)
                    except:
                        pass
        #### Check the quality of the signal and label it- 0: good , 1: corrupted
        ## the quality criteras are adopted from paper "Analysis of PCG signals using quality assessment and homomorphic filters for localization and classification of heart sounds"
        coeffs = pywt.wavedec(audio, 'db1', level=2)
        cA2, cD2, cD1 = coeffs  # cA2 is the approximation coefficients at level 2
        if calculate_rmssd(cA2)> 0.1 or zero_crossing_ratio(cA2)>0.3 or ratio_of_normal_peaks(cA2, fs )>0.5: # threshold is coming from a paper 
            ret['label']= 1 # signal is corrupted      
        else:
            ret['label']= 0

        # filtering the audio by a band pass filter 
        audio = bandpass_filter(audio, fs)
        # standardization the audio data 
        audio = (audio - np.mean(audio))/ np.std(audio)
        
        if self.resample: 
            # Compute the spectrogram of the audio signal to the aim of resampling 
            audio = np.asfortranarray(audio)
            D = np.abs(librosa.stft(audio))
            # Find the frequency bin with the maximum energy
            max_bin = np.argmax(D, axis=0)

            # Convert the frequency bin to Hertz
            frequencies = librosa.fft_frequencies(sr=fs)
            highest_frequency = np.max(frequencies[max_bin]) # getting the highest frequency in the spectrum of the signal        ret['audio'] = librosa.resample(data['audio'], orig_sr=data['sr'], target_sr=highest_frequency*2)
            # print("highest_frequency", highest_frequency)
            ret['audio'] = librosa.resample(audio, orig_sr= fs, target_sr=highest_frequency*2) # resampling procudure 
            ret['sr'] = highest_frequency*2
        else:
            ret['audio'] = audio
            ret['sr'] = fs
        ret['murmur'], ret['outcome'] = self.demo_data[self.demo_data['Patient ID']==self.subject_ids[index]][['Murmur', 'Outcome']].iloc[0]
        return ret
        
# Run a Sample!
path_demo = "../../Datasets/physionet.org/files/circor-heart-sound/1.0.3/training_data.csv"
path_sig = "../../Datasets/physionet.org/files/circor-heart-sound/1.0.3/training_data_wav"
data_sample = Data_loader(path_demo, path_sig, resample= False) # set resample to False and repeat 
data = data_sample.get_item(10)

print(data_sample.len())



# data_list = []
# SR =[]
# labels= []
sample_rate = 4000
for index in range(data_sample.len()):
    data_item = data_sample.get_item(index)
    if data_item['label'] == 0:
        output_file = f'/home/ainazj1/psanjay_ada/users/ainazj1/Datasets/physionet.org/files/circor-heart-sound/1.0.3/training_data_diffwave/output_{index}.wav'
        sf.write(output_file, data_item['audio'], sample_rate)

# # Process all instances in the dataset and collect data in a list
# data_list = []
# SR =[]
# labels= []
# for index in range(data_sample.len()):
#     data_item = data_sample.get_item(index)
#     data_list.append(data_item['audio'])
#     SR.append(data_item['sr'])
#     labels.append(data_item['label'])


# # Create a DataFrame from the collected data 
# pd.DataFrame(data_list).to_csv("PCG_timeseris_normal_processed_NoResample.csv")
# # pd.DataFrame(SR).to_csv('PCG_timeseres-SRs.csv')
# pd.DataFrame(labels).to_csv('PCG_timeseries_labels.csv', index= False)