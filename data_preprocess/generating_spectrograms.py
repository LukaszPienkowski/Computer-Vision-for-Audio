import os
import hashlib
import librosa
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image
from .audio_utils import generate_mel_spectrogram, save_spectrogram


def process_and_save_spectrograms(input_dir, output_dir):
    """
    Reads audio files from an input directory, divides them into 4-second segments,
    generates normalized Mel spectrograms for each segment, and saves them as PNG images.

    Args:
        input_dir (str): Path to the directory containing input audio files (e.g., .wav, .mp3, .flac).
        output_dir (str): Path to the directory where the resulting spectrogram images will be saved.

    Details:
        - Scans the input directory for supported audio formats.
        - Groups files by speaker ID based on the filename prefix (e.g., '123' from '123_1.wav').
        - Divides the audio into consecutive 4-second segments. Discards any leftover audio shorter than 4 seconds.
        - Converts each segment into a normalized Mel spectrogram in decibels (scaled 0-1).
        - Names output images sequentially per speaker (e.g., '{speaker_id}_{index}.png') and saves them.
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


def remove_duplicate_spectrograms(output_dir):
    """
    Removes physically duplicate spectrogram PNG files from a directory using MD5 hashing.

    Computes the MD5 hash of every PNG file in the directory. For each set of files
    sharing the same hash, keeps the first occurrence and deletes all subsequent duplicates.
    Augmented files (containing '_aug_' in the name) are skipped.

    Args:
        output_dir (str): Path to the directory to scan for duplicate spectrogram images.

    Returns:
        int: Number of duplicate files deleted.
    """
    files = [f for f in os.listdir(output_dir) if f.endswith('.png') and '_aug_' not in f]

    hashes = {}
    deleted = 0

    for fname in files:
        fpath = os.path.join(output_dir, fname)
        with open(fpath, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        if file_hash in hashes:
            os.remove(fpath)
            deleted += 1
        else:
            hashes[file_hash] = fpath

    print(f"Removed {deleted} duplicate spectrograms from {output_dir}.")
    return deleted


def remove_silence_from_spectrograms(output_dir, threshold=0.05, cut_end_percent=0.1):
    """
    Applies silence removal and end-trimming to all original spectrogram PNG files in a directory.

    For each image, silent time-frame columns are detected by checking whether the maximum
    pixel intensity in a column falls below a threshold. Those columns are dropped. Additionally,
    a percentage of the image from the right (end of recording) is removed to eliminate fade-out
    artefacts. The cleaned image overwrites the original file in-place.
    Augmented files (containing '_aug_' in the name) are skipped.

    Args:
        output_dir (str): Path to the directory containing spectrogram PNG images to process.
        threshold (float): Minimum normalised pixel intensity (0-1) for a column to be
                           considered non-silent (default: 0.05).
        cut_end_percent (float): Fraction of the image width to remove from the right end
                                 (default: 0.1, i.e. 10%).

    Returns:
        int: Number of files that were entirely silent and deleted.
    """
    files = [f for f in os.listdir(output_dir) if f.endswith('.png') and '_aug_' not in f]

    removed_silent = 0

    for fname in files:
        fpath = os.path.join(output_dir, fname)
        try:
            img = Image.open(fpath)
            img_arr = np.array(img)
            img_gray = np.array(img.convert('L')) / 255.0

            col_max = img_gray.max(axis=0)
            non_silent_mask = col_max > threshold

            if not np.any(non_silent_mask):
                os.remove(fpath)
                removed_silent += 1
                continue

            # Drop silent columns
            if len(img_arr.shape) == 3:
                img_clean = img_arr[:, non_silent_mask, :]
            else:
                img_clean = img_arr[:, non_silent_mask]

            # Trim the end
            w = img_clean.shape[1]
            new_w = int(w * (1 - cut_end_percent))
            img_clean = img_clean[:, :new_w] if len(img_clean.shape) == 2 else img_clean[:, :new_w, :]

            Image.fromarray(img_clean).save(fpath)

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    print(f"Silence removal complete for {output_dir}. {removed_silent} fully-silent files deleted.")
    return removed_silent


def main():
    classes = ["class_0", "class_1"]
    base_input_dir = "data"
    base_output_dir = "spectrograms"

    for cls in classes:
        input_dir = os.path.join(base_input_dir, cls)
        output_dir = os.path.join(base_output_dir, cls)

        process_and_save_spectrograms(input_dir, output_dir)

        print(f"Post-processing {output_dir}...")
        remove_duplicate_spectrograms(output_dir)
        remove_silence_from_spectrograms(output_dir)

    print("Spectrogram generation complete.")


if __name__ == "__main__":
    main()
