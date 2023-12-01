# this script checks if there is any empty data file in our set of wave files.
from scipy.io import wavfile
import os 
from glob import glob

def is_wav_file_empty(filename):
    try:
        sr, audio = wavfile.read(filename)
        return audio.size == 0
    except ValueError as e:
        print(f"Error reading {filename}: {e}")
        return True  # or handle differently depending on your use case

# paths = os.listdir('/home/ainazj1/psanjay_ada/users/ainazj1/diffwave/wave_processed')
directory = '/home/ainazj1/psanjay_ada/users/ainazj1/diffwave/wave_processed' 
paths = glob(os.path.join(directory, '*.wav'))

# i =1
r =[]
for p in paths:
    # print(p)
    # f = os.path.join(p, f'output_{i}.wav')
    r.append(is_wav_file_empty(p))
print(sum(r))
print(any(r))
# filename = '/home/ainazj1/psanjay_ada/users/ainazj1/diffwave/wave_processed/output_100.wav'
# if is_wav_file_empty(filename):
#     print(f"The file {filename} is empty.")
# else:
#     print(f"The file {filename} contains audio data.")