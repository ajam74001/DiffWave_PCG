import numpy as np
from scipy.signal import butter, filtfilt
# the frequency range of normal PCG signals are 40-200 : so we want to keep this frequency range of the signals 
def bandpass_filter(signal, fs,  lowcut=20, highcut=500, order=1): 
    """
    Apply a bandpass filter to the signal.
    
    Parameters:
    - signal: Input signal.
    - lowcut: Low cutoff frequency.
    - highcut: High cutoff frequency.
    - fs: Sampling rate.
    - order: Order of the filter.
    
    Returns:
    - Filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# Sample usage:
# fs = 1000  # Sample rate, for example, 1000 Hz. You need to replace this with your actual sampling rate.
# raw_signal = np.random.randn(fs * 10)  # Replace with your actual signal data.
# filtered_signal = bandpass_filter(raw_signal, lowcut=20, highcut=400, fs=fs)

