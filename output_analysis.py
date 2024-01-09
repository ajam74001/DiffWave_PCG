import numpy as np
import librosa 
import os 
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd 


dir = '/home/ainazj1/psanjay_ada/users/ainazj1/Datasets/physionet.org/files/circor-heart-sound/1.0.3/output_diffwave'

paths = glob(os.path.join(dir, '*.wav'))
data = []
for p in paths:
    audio, fs = librosa.load(p, mono=True, sr=None)
    data.append(np.array(audio))


print(pd.DataFrame(data).shape)
pd.DataFrame(data).to_csv('/home/ainazj1/psanjay_ada/users/ainazj1/Datasets/synth_normal_diffwave.csv', index=False)
