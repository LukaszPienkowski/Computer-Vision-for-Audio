import os
import subprocess
import sys

def has_files(directory, extensions):
    """Check if a directory contains any files with the given extensions."""
    if not os.path.exists(directory):
        return False
    for f in os.listdir(directory):
        if f.endswith(extensions):
            return True
    return False

def main():
    print("=== Pipeline Orchestrator ===")
    
    # 1. Check audio data
    data_exists = has_files("data/class_0", ('.wav', '.mp3', '.flac')) and \
                  has_files("data/class_1", ('.wav', '.mp3', '.flac'))
    
    if not data_exists:
        print("\n[1/3] Audio data not found. Running get_data.py...")
        subprocess.run([sys.executable, "-m", "data_preprocess.get_data"], check=True)
    else:
        print("\n[1/3] Audio data found. Skipping download.")

    # 2. Check spectrograms
    specs_exist = has_files("spectrograms/class_0", ('.png',)) and \
                  has_files("spectrograms/class_1", ('.png',))
    
    if not specs_exist:
        print("\n[2/3] Spectrograms not found. Running generating_spectrograms.py...")
        subprocess.run([sys.executable, "-m", "data_preprocess.generating_spectrograms"], check=True)
    else:
        print("\n[2/3] Spectrograms found. Skipping generation.")

    # 3. Check augmentation
    # augment_spectrograms.py inherently checks the counts and skips if already balanced.
    print("\n[3/3] Checking if spectrogram augmentation is needed to balance classes...")
    subprocess.run([sys.executable, "-m", "data_preprocess.augment_spectrograms"], check=True)

    print("\n=== Pipeline Execution Complete ===")

if __name__ == "__main__":
    main()
