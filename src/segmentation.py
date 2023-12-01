import numpy as np
import pandas as pd
import math
import random
from scipy.signal import find_peaks
from utilities import normalize_signal


#TODO: Use only the signals which passed the quality assesment tests - label 0
# real = pd.read_csv('/userfiles/ajamshidi18/PCG_timeseris_normal_processed.csv').drop('Unnamed: 0', axis =1)
# labels = pd.read_csv('/userfiles/ajamshidi18/PCG_timeseries_labels.csv')
# indices_to_filter = labels.index[labels['0'] == 1].tolist() # filtering the corrupted signals 
# real = real.drop(indices_to_filter)
def segment(real1):
    # final_data = [] 
    win_len =100
    # for indx in range(data.shape[0]):
    # real1  = data.dropna()
    real1 = real1[int(0.1*len(real1)):-int(0.1*len(real1)), ]
    peaks, info = find_peaks(real1, height= 2) # sounds good!
    real_norm = normalize_signal(real1)
    segments=[]
    for i,p in enumerate(peaks):
            if info['peak_heights'][i]<4:
                if real_norm[p-win_len:p+10]: # if it is non empty 
                    segments.append(real_norm[p-win_len:p+10])

    # final_data.append(segments)
    return segments
    # flattened_data = [item for sublist in final_data for item in sublist]
# pd.DataFrame(flattened_data).to_csv("/userfiles/ajamshidi18/PCG_timeseris_normal_final_P_P.csv")