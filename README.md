# Computer Vision for classifing the audio

This repository contains tools for fetching audio datasets and generating spectrograms for computer vision tasks.

## `get_data.py`

The `get_data.py` script downloads audio files from the Hugging Face Hub (defaulting to the `facebook/voxpopuli` dataset).

### Functionality:
- **Hugging Face Integration**: Streams the dataset using the `datasets` library. Requires a valid Hugging Face token (`HF_TOKEN` in a `.env` file).
- **Data Filtering**: Ensures only one audio file per speaker is downloaded and filters out recordings shorter than 25 seconds.
- **Class Splitting**: Automatically divides the downloaded data into two classes (15% in `class_1` and 85% in `class_0`) to simulate an imbalanced dataset or specific data distribution.
- **Local Data Integration**: Automatically checks for a `my_records` folder. If found, any locally recorded `.wav` files are copied into `data/class_0`. This injects local microphone background noise into the base dataset to eliminate domain shift bias.
- **File Output**: Cleans and recreates the data directories, saving the `.wav` files into `data/class_0` and `data/class_1`.

## `generating_spectrograms.py`

The `generating_spectrograms.py` script converts downloaded audio files into Mel spectrogram images for use in computer vision tasks. After generation, it automatically applies post-processing to clean the dataset.

### Functionality:
- **process_and_save_spectrograms**: The core function that drives the spectrogram pipeline.
  - **Speaker Grouping**: Identifies speaker IDs from filenames (e.g., `123` from `123_1.wav`) and groups all related audio files.
  - **Audio Splitting**: Processes files in order, dividing them into consecutive 4-second segments, and discarding any leftovers.
  - **Spectrogram Generation**: Converts each segment into a normalized Mel spectrogram using the `audio_utils` module.
  - **Sequential Naming & Output**: Saves each segment as a PNG image in the specified output directory, named sequentially for each speaker (e.g., `{speaker_id}_{index}.png`).
- **remove_duplicate_spectrograms**: Scans a spectrogram directory and removes physically identical PNG files using MD5 hashing. Keeps the first occurrence of each unique image and deletes the rest. Augmented files (`_aug_`) are skipped.
- **remove_silence_from_spectrograms**: Cleans each spectrogram image in-place by dropping silent time-frame columns (columns whose maximum pixel intensity falls below a configurable threshold) and trimming a configurable percentage from the right end of the image to remove fade-out artefacts. Fully silent images are deleted. Augmented files (`_aug_`) are skipped.

## `audio_utils.py`

The `audio_utils.py` script contains core helper functions for processing audio signals and rendering them as spectrogram images. 

### Functionality:
- **generate_mel_spectrogram**: Processes raw audio time-series data using `librosa` and returns a normalized (0 to 1) Mel spectrogram in decibels.
- **save_spectrogram**: Saves the normalized spectrogram array to the filesystem as a PNG image using a configurable colormap (default: `magma`).
- **spectrogram_to_image**: Converts the normalized array directly into a resized PIL Image, which is useful for real-time inference or GUI applications.
- **time_mask**: Applies a random vertical black bar to the spectrogram image to simulate missing audio frames over a specific duration (Data Augmentation).
- **freq_mask**: Applies a random horizontal black bar to the spectrogram image to simulate missing frequency bands (Data Augmentation).
- **add_noise_to_image**: Adds a specified amount of random Gaussian noise directly to the spectrogram pixels (Data Augmentation).

## `augment_spectrograms.py`

The `augment_spectrograms.py` script balances the spectrogram dataset by generating augmented image copies of the minority class using computer vision techniques.

### Functionality:
- **augment_added_data**: Applies image-level augmentations to spectrogram PNG files in a given directory. Generates `factor` augmented copies per original image using randomly chosen transformations. Augmented files are marked with an `_aug_` infix to prevent re-processing.
- **main**: Compares the image counts of `class_0` and `class_1` in the `spectrograms/` directory. If `class_1` is underrepresented, generates the exact number of additional augmented samples needed, sampling randomly from existing original Class 1 spectrograms. Available transformations:
  - **Time Mask**: Applies a random vertical black bar to simulate missing audio frames.
  - **Frequency Mask**: Applies a random horizontal black bar to simulate missing frequency bands.
  - **Gaussian Noise**: Adds random pixel noise to simulate measurement variability.

## `main.py`

The `main.py` script is the top-level pipeline orchestrator. Run this file to execute the entire preprocessing workflow from raw data to a balanced spectrogram dataset.

### Functionality:
- **Audio Data Check**: Verifies whether audio files exist in `data/class_0` and `data/class_1`. If not, automatically runs `get_data.py` to download them.
- **Spectrogram Check**: Verifies whether spectrogram images exist in `spectrograms/class_0` and `spectrograms/class_1`. If not, automatically runs `generating_spectrograms.py`.
- **Augmentation Check**: Runs `augment_spectrograms.py`, which self-checks class balance and only generates augmented images if needed.

## `model.py`

The `model.py` script serves as the central definition file for the project's Neural Network architectures and includes standalone training routines for establishing base models from the generated spectrograms.

### Functionality:
- **`CustomCNN`**: A lightweight PyTorch Convolutional Neural Network architecture designed for classifying audio spectrogram images. Features three convolutional layers with batch normalization and max pooling.
- **`DeepCNN`**: A deeper, more complex architecture tailored for difficult classifications. Features five convolutional layers with batch normalization, higher dropout rates, and more parameters to prevent overfitting.
- **`evaluate_detailed`**: Evaluates a trained PyTorch model against a given dataloader and calculates detailed performance metrics including Accuracy, Precision, Recall, and F1-Score.
- **`train_model`**: Drives the standalone training loop. Trains a specified PyTorch model architecture, evaluates its performance on the test set, saves the resulting optimal weights to the `models/` directory, and returns the evaluation metrics.
- **Standalone Execution**: When run directly, the script will automatically train both the `CustomCNN` and `DeepCNN` models on the data in the `spectrograms/` directory and display a comparative performance table and bar chart.

## `fine_tune_model.py`

The `fine_tune_model.py` script adapts a pre-trained base model to recognize a specific user's voice by fine-tuning the neural network weights on new custom data.

### Functionality:
- **`preprocess_added_data`**: Reads custom audio files (`.wav`, `.mp3`, `.flac`) from a directory, chunks them into 4-second segments, and generates spectrograms in the `spectrograms_added/` folder.
- **`fine_tune`**: Core fine-tuning loop. Mixes the user's new custom spectrograms (`class_1`) with base dataset spectrograms (`class_0`) to prevent catastrophic forgetting. Uses early stopping and automatically detects and loads the best pre-trained weights to adapt to the new voice profile.

## `gui_app.py`

The `gui_app.py` script is a fully featured graphical application built with `customtkinter`. It serves as the primary front-end for users to interact with the trained models.

### Functionality:
- **Microphone Integration**: Allows users to record their voice directly from their microphone to test the model's live inference accuracy.
- **Live Inference**: Automatically chunks live audio or selected files, processes them into spectrogram images using the `magma` colormap, and passes them to the loaded neural network for real-time classification.
- **Fine-Tuning Integration**: Provides a visual workflow to select audio files, trigger the `fine_tune_model.py` backend script, and automatically reload the updated weights for immediate testing.

---

## `EDA_spectrograms.ipynb`

This Jupyter Notebook focuses on the Exploratory Data Analysis (EDA) of the generated image dataset. It allows for deep visual inspection of the spectrograms to ensure data quality before model training.

### Functionality:
- **Data Cleaning Verification**: Checks for duplicate images using MD5 hashing to ensure the dataset is perfectly clean.
- **Mean Spectrogram Analysis**: Computes and visualizes the average macro-level visual features for each class, helping to identify consistent acoustic differences between Class 0 and Class 1.
- **Pixel Intensity Distributions**: Plots density histograms of pixel intensities (loudness/energy) to uncover systematic biases, such as one class being consistently louder than the other.

---

## `Model_Driven_EDA.ipynb`

This Jupyter Notebook performs Model-Driven Exploratory Data Analysis. It leverages the trained Convolutional Neural Networks (from `model.py`) to analyze the dataset from the model's perspective, mapping how the AI interprets the audio.

### Functionality:
- **Feature Extraction**: Uses a custom wrapper to extract high-dimensional latent embeddings (the raw features extracted just before the final classification layer) for any given spectrogram.
- **Prediction Confidence Analysis**: Evaluates the model's certainty on individual samples to identify difficult or ambiguous audio clips.
- **Cosine Similarity Search**: Calculates the cosine distance between extracted embeddings to find and visualize the most "similar" audio samples in the latent space, revealing what acoustic features the model groups together.