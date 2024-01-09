import numpy as np
# import torch
# import torchaudio as T
# import torchaudio.transforms as TT
# from argparse import ArgumentParser
# from concurrent.futures import ProcessPoolExecutor
# from glob import glob
# from tqdm import tqdm
import librosa 
import os 
# directory_path = "/home/ainazj1/psanjay_ada/users/ainazj1/Datasets/physionet.org/files/circor-heart-sound/1.0.3/training_data_wav"
# filenames = os.listdir(directory_path)
# print(filenames)
# sr_uniqs= []
# for path in filenames:
#     # audio, fs = T.load(path)
#     audio , fs = librosa.load(path, mono=True, sr=None)
#     sr_uniqs.append(fs)

# print(set(sr_uniqs))

directory_path = "/home/ainazj1/psanjay_ada/users/ainazj1/Datasets/physionet.org/files/circor-heart-sound/1.0.3/training_data_wav"
filenames = os.listdir(directory_path)

sr_uniqs = []

for filename in filenames:
    # Create the full path to the file
    path = os.path.join(directory_path, filename)
    
    # Load the audio file using librosa
    audio, fs = librosa.load(path, mono=True, sr=None)
    
    # Append the sample rate to the sr_uniqs list
    sr_uniqs.append(fs)

print(set(sr_uniqs))
print(len(sr_uniqs))

# 4000 HZ