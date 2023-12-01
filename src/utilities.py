import numpy as np 
import pandas as pd
from scipy.signal import find_peaks

def normalize_signal(signal):
    min_val = min(signal)
    max_val = max(signal)
    normalized = [2 * ((x - min_val) / (max_val - min_val)) - 1 for x in signal]
    return normalized


def calculate_rmssd(signal):
        successive_differences = np.diff(signal)
        squared_differences = successive_differences**2
        mean_squared_diff = np.mean(squared_differences)
        rmssd = np.sqrt(mean_squared_diff)
        return rmssd

def zero_crossing_ratio(signal):
    """
    Calculate the ratio of zero crossings to the length of the signal.
    
    :param signal: Array-like sequence of signal samples.
    :return: The zero crossing ratio.
    """
    # Shift the signal by one and multiply by the original signal to find crossings
    zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
    # Calculate the ratio
    ratio = len(zero_crossings) / len(signal)
    
    return ratio

def ratio_of_normal_peaks(signal, sampling_rate, window_duration_ms=2200, peak_range=(2, 10)):
    """
    Calculate the ratio of windows with a normal number of peaks to the total number of windows.
    
    :param signal: Array-like sequence of signal samples.
    :param sampling_rate: Sampling rate of the signal in Hz.
    :param window_duration_ms: Duration of each window in milliseconds.
    :param peak_range: Tuple specifying the normal range of peaks (min_peaks, max_peaks).
    :return: The ratio of windows with normal peaks.
    """
    window_size = int(sampling_rate * (window_duration_ms / 1000.0))
    num_windows = len(signal) // window_size
    normal_peak_windows = 0

    for i in range(num_windows):
        start_index = i * window_size
        end_index = start_index + window_size
        window = signal[start_index:end_index]
        peaks, _ = find_peaks(window)
        if peak_range[0] <= len(peaks) <= peak_range[1]:
            normal_peak_windows += 1

    # print(num_windows)
    try:
        normal_peak_ratio = normal_peak_windows / num_windows
    except ZeroDivisionError:
        normal_peak_ratio = 1
    
    return normal_peak_ratio