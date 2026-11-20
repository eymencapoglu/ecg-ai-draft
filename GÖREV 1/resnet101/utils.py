
import numpy as np
import scipy.signal as signal

def clean_signal(data, fs=500, lowcut=0.5, highcut=45.0, order=4):
    """
    Applies Bandpass filter to remove noise and baseline wander.
    data: (channels, length) numpy array
    fs: sampling rate
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Check for valid critical frequencies
    if low <= 0 or high >= 1:
        # Fallback if cuts are out of bounds for the sampling rate, though typical ECG is fine.
        return data
        
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter along time axis (axis=1)
    cleaned = signal.filtfilt(b, a, data, axis=1)
    return cleaned
