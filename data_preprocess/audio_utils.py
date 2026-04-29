import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def generate_mel_spectrogram(y, sr, n_mels=128, fmax=8000):
    """
    Generates a normalized Mel spectrogram from an audio signal.
    
    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of `y`.
        n_mels (int): Number of Mel bands to generate (default: 128).
        fmax (int): Highest frequency (in Hz) (default: 8000).
        
    Returns:
        np.ndarray: A normalized Mel spectrogram in decibels (dB), scaled between 0 and 1.
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    min_db, max_db = S_dB.min(), S_dB.max()
    if max_db > min_db:
        S_dB_norm = (S_dB - min_db) / (max_db - min_db)
    else:
        S_dB_norm = np.zeros_like(S_dB)
        
    return S_dB_norm

def save_spectrogram(S_dB_norm, output_path, cmap='magma'):
    """
    Saves a normalized spectrogram as a PNG image.
    
    Args:
        S_dB_norm (np.ndarray): Normalized Mel spectrogram (e.g., from `generate_mel_spectrogram`).
        output_path (str): File path where the image will be saved.
        cmap (str): Colormap to use for saving the image (default: 'magma').
    """
    plt.imsave(output_path, S_dB_norm, cmap=cmap)

def spectrogram_to_image(S_dB_norm, size=(128, 128)):
    """
    Converts a normalized spectrogram array to a resized PIL Image.
    Useful for GUI and real-time inference.
    
    Args:
        S_dB_norm (np.ndarray): Normalized Mel spectrogram.
        size (tuple): Desired (width, height) of the output image (default: (128, 128)).
        
    Returns:
        PIL.Image.Image: The resized PIL Image representation of the spectrogram.
    """
    img_data = (S_dB_norm * 255).astype(np.uint8)
    img = Image.fromarray(img_data)
    if size:
        img = img.resize(size)
    return img

def time_mask(img, max_mask_width=15):
    """
    Applies a vertical black bar (time mask) to the spectrogram image.
    
    Args:
        img (PIL.Image): The input spectrogram image.
        max_mask_width (int): Maximum width of the mask in pixels.
        
    Returns:
        PIL.Image: The masked image.
    """
    import random
    img_arr = np.array(img)
    mask_width = random.randint(1, max_mask_width)
    start = random.randint(0, max(1, img_arr.shape[1] - mask_width))
    # Fill with black
    if len(img_arr.shape) == 3:
        img_arr[:, start:start+mask_width, :] = 0
    else:
        img_arr[:, start:start+mask_width] = 0
    return Image.fromarray(img_arr)

def freq_mask(img, max_mask_width=15):
    """
    Applies a horizontal black bar (frequency mask) to the spectrogram image.
    
    Args:
        img (PIL.Image): The input spectrogram image.
        max_mask_width (int): Maximum width of the mask in pixels.
        
    Returns:
        PIL.Image: The masked image.
    """
    import random
    img_arr = np.array(img)
    mask_width = random.randint(1, max_mask_width)
    start = random.randint(0, max(1, img_arr.shape[0] - mask_width))
    if len(img_arr.shape) == 3:
        img_arr[start:start+mask_width, :, :] = 0
    else:
        img_arr[start:start+mask_width, :] = 0
    return Image.fromarray(img_arr)

def add_noise_to_image(img, factor=0.03):
    """
    Adds Gaussian noise to the spectrogram image.
    
    Args:
        img (PIL.Image): The input spectrogram image.
        factor (float): The noise factor/intensity.
        
    Returns:
        PIL.Image: The noisy image.
    """
    img_arr = np.array(img).astype(np.float32)
    noise = np.random.randn(*img_arr.shape) * 255 * factor
    img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img_arr)
