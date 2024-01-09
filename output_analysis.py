import numpy as np
import librosa 
import os 
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd 
# path = '/home/ainazj1/psanjay_ada/users/ainazj1/output.wav'
# audio, fs = librosa.load(path, mono=True, sr=None)
# # print(len(audio[3000:4000]))
# plt.title('Synthetic')
# plt.plot(audio[2500:4000])
# plt.xticks([0, 500, 1000, 1500])
# plt.xlabel('Time')
# plt.ylabel('Amp')
# plt.savefig('synth_diffwave.png')

dir = '/home/ainazj1/psanjay_ada/users/ainazj1/Datasets/physionet.org/files/circor-heart-sound/1.0.3/output_diffwave'
# list_dir = os.listdir(dir)
paths = glob(os.path.join(dir, '*.wav'))
data = []
for p in paths:
    audio, fs = librosa.load(p, mono=True, sr=None)
    data.append(np.array(audio))
# print(np.array(data).shape)
# print(data)
print(pd.DataFrame(data).shape)
pd.DataFrame(data).to_csv('/home/ainazj1/psanjay_ada/users/ainazj1/Datasets/synth_normal_diffwave.csv', index=False)
