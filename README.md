# Computer Vision for classifing the audio

This repository contains tools for fetching audio datasets and generating spectrograms for computer vision tasks.

## `get_data.py`

The `get_data.py` script downloads audio files from the Hugging Face Hub (defaulting to the `facebook/voxpopuli` dataset).

### Functionality:
- **Hugging Face Integration**: Streams the dataset using the `datasets` library. Requires a valid Hugging Face token (`HF_TOKEN` in a `.env` file).
- **Data Filtering**: Ensures only one audio file per speaker is downloaded and filters out recordings shorter than 25 seconds.
- **Class Splitting**: Automatically divides the downloaded data into two classes (15% in `class_1` and 85% in `class_0`) to simulate an imbalanced dataset or specific data distribution.
- **File Output**: Cleans and recreates the data directories, saving the `.wav` files into `data/class_0` and `data/class_1`.

## `generating_spectrograms.py`

The `generating_spectrograms.py` script converts downloaded audio files into Mel spectrogram images for use in computer vision tasks.

### Functionality:
- **process_and_save_spectrograms**: The core function that drives the spectrogram pipeline.
  - **Speaker Grouping**: Identifies speaker IDs from filenames (e.g., `123` from `123_1.wav`) and groups all related audio files.
  - **Audio Splitting**: Processes files in order, dividing them into consecutive 4-second segments, and discarding any leftovers.
  - **Spectrogram Generation**: Converts each segment into a normalized Mel spectrogram using the `audio_utils` module.
  - **Sequential Naming & Output**: Saves each segment as a PNG image in the specified output directory, named sequentially for each speaker (e.g., `{speaker_id}_{index}.png`).

## `audio_utils.py`

The `audio_utils.py` script contains core helper functions for processing audio signals and rendering them as spectrogram images. 

### Functionality:
- **generate_mel_spectrogram**: Processes raw audio time-series data using `librosa` and returns a normalized (0 to 1) Mel spectrogram in decibels.
- **save_spectrogram**: Saves the normalized spectrogram array to the filesystem as a PNG image using a configurable colormap (default: `magma`).
- **spectrogram_to_image**: Converts the normalized array directly into a resized PIL Image, which is useful for real-time inference or GUI applications.
- **time_mask**: Applies a random vertical black bar to the spectrogram image to simulate missing audio frames over a specific duration (Data Augmentation).
- **freq_mask**: Applies a random horizontal black bar to the spectrogram image to simulate missing frequency bands (Data Augmentation).
- **add_noise_to_image**: Adds a specified amount of random Gaussian noise directly to the spectrogram pixels (Data Augmentation).