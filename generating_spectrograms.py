import os
import librosa
import numpy as np
from pathlib import Path
from collections import defaultdict
from audio_utils import generate_mel_spectrogram, save_spectrogram

def process_and_save_spectrograms(input_dir, output_dir):
    """
    Reads audio files from an input directory, divides them into 4-second segments,
    generates normalized Mel spectrograms for each segment, and saves them as PNG images.
    
    Args:
        input_dir (str): Path to the directory containing input audio files (e.g., .wav, .mp3, .flac).
        output_dir (str): Path to the directory where the resulting spectrogram images will be saved.
        
    Details:
        - Groups files by speaker ID based on the filename prefix (e.g., '123_1.wav' -> '123').
        - Discards any leftover audio that is less than the 4-second segment duration.
        - Names output images sequentially per speaker (e.g., '{speaker_id}_{index}.png').
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    supported_extensions = ('.wav', '.mp3', '.flac')
    files = [f for f in os.listdir(input_dir) if f.endswith(supported_extensions)]
    
    speaker_files = defaultdict(list)
    for f in files:
        speaker_id = f.split('_')[0]
        speaker_files[speaker_id].append(f)
    
    print(f"Found {len(speaker_files)} speakers in {input_dir}...")

    segment_duration = 4.0

    for speaker_id, file_list in speaker_files.items():
        file_list.sort()
        segment_index = 1
        
        for file_name in file_list:
            audio_path = os.path.join(input_dir, file_name)
            
            try:
                y, sr = librosa.load(audio_path)
                samples_per_segment = int(segment_duration * sr)
                num_segments = len(y) // samples_per_segment
                
                if num_segments == 0:
                    continue

                for i in range(num_segments):
                    start = i * samples_per_segment
                    end = start + samples_per_segment
                    y_segment = y[start:end]

                    S_dB_norm = generate_mel_spectrogram(y_segment, sr)
                    
                    output_path = os.path.join(output_dir, f"{speaker_id}_{segment_index}.png")
                    save_spectrogram(S_dB_norm, output_path)
                    segment_index += 1
                
            except Exception as e:
                print(f"Error in a file {file_name}: {e}")

if __name__ == "__main__":
    classes = ["class_0", "class_1"]
    base_input_dir = "data"
    base_output_dir = "spectrograms"

    for cls in classes:
        input_dir = os.path.join(base_input_dir, cls)
        output_dir = os.path.join(base_output_dir, cls)
        
        process_and_save_spectrograms(input_dir, output_dir)

    print("Done")
