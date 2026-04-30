# Computer Vision for classifing the audio

This repository contains tools for fetching audio datasets and generating spectrograms for computer vision tasks.

## `get_data.py`

The `get_data.py` script downloads English audio from the [`facebook/voxpopuli`](https://huggingface.co/datasets/facebook/voxpopuli) Hugging Face dataset and organises it into two classes by speaker identity. Class 0 is built from VoxPopuli speakers (large, diverse "world" set). Class 1 target-speaker recordings are sourced from the same dataset and the ones placed in `my_records/`.

### Functionality:
- **Hugging Face Integration**: Streams data using the `datasets` library. Requires a `HF_TOKEN` in a `.env` file.
- **Data Filtering**: Keeps one recording per speaker and discards files shorter than 25 seconds.
- **Class Split by Speaker**: Downloads up to 500 unique speakers into `class_0`. At the end, randomly selects 5–7 of them and moves their files to `class_1`. Both classes share the same audio domain, forcing the model to learn voice characteristics rather than audio format differences.
- **Local Data Integration**: Copies `.wav` files from `my_records/` into `data/class_0` to inject local microphone noise and reduce domain shift.
- **File Output**: Cleans and recreates `data/class_0` and `data/class_1` on each run.

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

The `augment_spectrograms.py` script applies image-level augmentation to the minority class spectrograms. The dataset is intentionally imbalanced (class_0 >> class_1), so augmentation is applied only to class_1 to improve its representation without forcing an artificial 1:1 balance.

### Functionality:
- **augment_added_data**: Applies image-level augmentations to spectrogram PNG files in a given directory. Generates `factor` augmented copies per original image. Augmented files are marked with an `_aug_` infix to prevent re-processing.
- **main**: Applies 2× augmentation to all original class_1 spectrograms. Reports counts for both classes and leaves class_0 untouched. Available transformations:
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

The `model.py` script defines the Neural Network architectures and contains training routines for establishing base models from the generated spectrograms.

### Functionality:
- **`speaker_level_split`**: Splits the dataset into train / val / test at the **speaker** level (70 / 15 / 15). All spectrogram segments from a single speaker land entirely in one partition, preventing data leakage.
- **`make_weighted_loader`**: Builds a `WeightedRandomSampler`-backed DataLoader that rebalances batches by inverse class frequency.
- **`CustomCNN`**: A lightweight CNN with three convolutional layers, batch normalisation, and max pooling.
- **`DeepCNN`**: A deeper architecture with five convolutional layers (two VGG-style blocks), higher dropout, and more parameters.
- **`evaluate_detailed`**: Evaluates a model on a DataLoader and returns Accuracy, Precision, Recall, F1-Score, **FAR** (False Acceptance Ratio), and **FRR** (False Rejection Ratio). Metrics are reported for train, val, and test splits separately.
- **`train_model`**: Runs the training loop with:
  - Class-weighted `CrossEntropyLoss` to handle class imbalance.
  - Early stopping monitored on the **validation** set (not the test set).
  - Choice of **Adam** (`lr=0.001`) or **SGD** (`lr=0.01`, momentum=0.9, weight_decay=1e-4, StepLR scheduler).
  - **Model serialisation**: Best weights are saved to `models/{name}_best.pth` (Adam) or `models/{name}_SGD_best.pth` (SGD) via `torch.save`. The best checkpoint is reloaded automatically before final evaluation.
  - Loss curve saved to `plots/` after each run.
- **Standalone Execution**: Trains `CustomCNN` and `DeepCNN` with both Adam and SGD (4 variants total), prints FAR/FRR for each split, and saves a comparative bar chart and CSV to `plots/`.

## `fine_tune_model.py`

The `fine_tune_model.py` script adapts a pre-trained base model to recognise additional target voices supplied by the end user.

### Functionality:
- **`preprocess_added_data`**: Reads audio files (`.wav`, `.mp3`, `.flac`) from `added_data/`, chunks them into 4-second segments, and saves Mel spectrograms to `spectrograms_added/class_1/`.
- **`augment_class1_specs`**: Applies 4× image augmentation (time mask, frequency mask, Gaussian noise) to the generated spectrograms to compensate for the small number of user-supplied samples.
- **`fine_tune`**: Mixes augmented class_1 spectrograms with up to 10× as many class_0 spectrograms from the base dataset to prevent catastrophic forgetting. Applies `WeightedRandomSampler` and class-weighted loss to handle the remaining imbalance. Trains with early stopping and saves the best weights.

## `gui_app.py`

The `gui_app.py` script is a fully featured graphical application built with `customtkinter`. It serves as the **final product** front-end for users to interact with the trained models. It fulfils the *"Jupyter-notebook-based program"* role described in the project brief — the decision to use a dedicated GUI app (rather than a notebook) was made to enable real-time microphone streaming, which is not reliably achievable inside a notebook kernel.

### Functionality:
- **Microphone Integration**: Allows users to record their voice directly from their microphone to test the model's live inference accuracy.
- **Live Inference**: Automatically chunks live audio or selected files, processes them into spectrogram images using the `magma` colormap, and passes them to the loaded neural network for real-time classification. Displays the predicted class and confidence level.
- **Fine-Tuning Integration**: Provides a visual workflow to select audio files, trigger the `fine_tune_model.py` backend script, and automatically reload the updated weights for immediate testing.

---

## `Visual_EDA.ipynb`

This Jupyter Notebook combines two visual exploratory analyses into a single runnable document.

### Part 1 — Background Noise Study
Inspects the influence of background noise by comparing spectrogram characteristics across three groups inside `spectrograms/class_0`:
- **VoxPopuli (clean)** — studio-quality European Parliament recordings.
- **Local mic (noisy)** — real-world laptop microphone recordings from `my_records/` (prefix `local_record_`).
- **class_1 target speakers** — shown as a reference baseline.

Three analyses are produced:
- **Mean Spectrogram Images**: pixel-wise average per group, revealing spectral shape differences caused by noise.
- **Pixel Intensity Distribution**: overlapping histograms with summary statistics (mean, std, median).
- **Per-Frame Energy Profile**: mean intensity across the time axis, exposing noise floors and fade patterns.

### Part 2 — Augmentation Visual Comparison
Displays a `(N × 4)` grid of original class_1 spectrograms alongside each augmentation type used during training:

| Original | Time Mask | Frequency Mask | Gaussian Noise |
|---|---|---|---|

All output plots are saved to `plots/`.

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

---

## `plots/` (generated)

All scripts that produce visualisations save their output here. This directory is created automatically on first run.

| File | Produced by |
|---|---|
| `model_comparison.png` / `.csv` | `model.py` — comparative bar chart and table for all 4 model variants |
| `{name}_{optimizer}_loss.png` | `model.py` — per-run loss curves (train vs val) |
| `noise_mean_spectrograms.png` | `Visual_EDA.ipynb` (Part 1) |
| `noise_intensity_distribution.png` | `Visual_EDA.ipynb` (Part 1) |
| `noise_energy_profile.png` | `Visual_EDA.ipynb` (Part 1) |
| `augmentation_comparison.png` | `Visual_EDA.ipynb` (Part 2) |

---

## Techniques & Functionalities Used

A brief overview of the key techniques implemented in this project, mapped to the course requirements. The project satisfies the **minimum four architectural/training techniques** required by the brief (items marked ✱ below correspond directly to the four chosen techniques from the requirement list):

| Requirement | Implementation |
|---|---|
| **Voice → Spectrogram preprocessing** | `generating_spectrograms.py` — 4-second Mel spectrogram segments via `librosa` |
| **Data cleaning** | Silence removal, duplicate detection (MD5), fade-out trimming in `generating_spectrograms.py` |
| **Speaker-level train/val/test split** | `speaker_level_split` in `model.py` — 70/15/15, no leakage across splits |
| **FAR / FRR metrics** | `evaluate_detailed` in `model.py` — reported for all three splits |
| ✱ **Data augmentation** *(req. a)* | Time mask, frequency mask, Gaussian noise — `audio_utils.py` & `augment_spectrograms.py`; silence removal & 4-second segment length also covered |
| ✱ **CNN architecture depth** *(req. b)* | `CustomCNN` (3-layer) vs. `DeepCNN` (5-layer VGG-style) — different number and size of layers |
| ✱ **Optimiser comparison** *(req. c)* | Adam vs. SGD (with StepLR scheduler, weight decay) — 4 variants trained and compared in `model.py` |
| ✱ **Batch normalisation** *(req. d)* | Applied after every convolutional layer in both `CustomCNN` and `DeepCNN` |
| **Dropout** | `CustomCNN`: 0.4; `DeepCNN`: 0.5 — reduces overfitting in both architectures |
| **Model serialisation** | Best weights saved to `models/*.pth` via `torch.save`; reloaded for evaluation and fine-tuning |
| **Class imbalance handling** | `WeightedRandomSampler` + class-weighted `CrossEntropyLoss` |
| **Transfer learning / fine-tuning** | `fine_tune_model.py` — adapts the base model to new speakers without catastrophic forgetting |
| **Adding a new person to Class 1** | End-to-end workflow: `fine_tune_model.py` → `gui_app.py` fine-tune button |
| **Live microphone inference** | `gui_app.py` — real-time recording, spectrogram conversion, and classification with confidence |
| **Background noise study** | `Visual_EDA.ipynb` Part 1 — mean spectrograms, intensity distributions, energy profiles |
| **Model-driven EDA** | `Model_Driven_EDA.ipynb` — latent embeddings, cosine similarity, confidence analysis |
| **Model comparison** | Bar chart + CSV in `plots/` generated by `model.py` |

---

## To Do

- [ ] Create report
- [ ] Create unit tests

