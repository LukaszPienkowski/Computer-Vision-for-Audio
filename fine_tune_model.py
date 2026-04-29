import os
import random
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from data_preprocess.audio_utils import generate_mel_spectrogram, save_spectrogram, time_mask, freq_mask, add_noise_to_image
from model import CustomCNN, DeepCNN

SUPPORTED_AUDIO = ('.wav', '.mp3', '.flac')

def preprocess_added_data(input_base, output_base):
    """
    Reads custom audio recordings (.wav, .mp3, .flac) from an input directory,
    splits them into uniform 4-second chunks to match the base dataset, and
    converts them into Mel spectrogram images.

    Args:
        input_base (str): The directory containing the raw user audio files.
        output_base (str): The directory where the generated spectrograms will be saved.
    
    Returns:
        int: Number of spectrogram images generated.
    """
    import librosa

    if not os.path.exists(input_base):
        print(f"Directory {input_base} does not exist.")
        return 0

    class1_dir = os.path.join(output_base, "class_1")
    if os.path.exists(class1_dir):
        shutil.rmtree(class1_dir)
    os.makedirs(class1_dir, exist_ok=True)

    count = 0
    for filename in os.listdir(input_base):
        if not filename.lower().endswith(SUPPORTED_AUDIO):
            continue

        audio_path = os.path.join(input_base, filename)
        try:
            y, sr = librosa.load(audio_path, sr=22050)

            chunk_length = 4 * sr
            if len(y) < chunk_length:
                print(f"  Skipping {filename}: too short (<4s).")
                continue

            base_name = os.path.splitext(filename)[0]
            for idx in range(len(y) // chunk_length):
                start = idx * chunk_length
                y_chunk = y[start:start + chunk_length]
                S_dB_norm = generate_mel_spectrogram(y_chunk, sr)
                out_path = os.path.join(class1_dir, f"{base_name}_{idx}.png")
                save_spectrogram(S_dB_norm, out_path)
                count += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Generated {count} spectrogram(s) from {input_base}.")
    return count


def augment_class1_specs(class1_dir, factor=4):
    """
    Applies image-level augmentations to all original class_1 spectrogram images.
    Generates `factor` augmented copies per original image using time mask, frequency
    mask, and Gaussian noise transforms to compensate for the small class_1 dataset.

    Args:
        class1_dir (str): Path to the class_1 spectrogram directory.
        factor (int): Number of augmented copies to produce per original image (default: 4).
    """
    augmentations = [
        ("tmask", lambda img: time_mask(img)),
        ("fmask",  lambda img: freq_mask(img)),
        ("noise",  lambda img: add_noise_to_image(img, factor=random.uniform(0.01, 0.05))),
    ]

    originals = [f for f in os.listdir(class1_dir) if f.endswith('.png') and '_aug_' not in f]
    print(f"Augmenting {len(originals)} original class_1 spectrograms (×{factor})...")

    for f in originals:
        src_path = os.path.join(class1_dir, f)
        base = f.rsplit('.', 1)[0]
        try:
            img = Image.open(src_path)
            for i in range(factor):
                aug_name, aug_fn = random.choice(augmentations)
                aug_img = aug_fn(img)
                aug_img.save(os.path.join(class1_dir, f"{base}_aug_{aug_name}_{i}.png"))
        except Exception as e:
            print(f"Error augmenting {f}: {e}")


class FixedLabelDataset(Dataset):
    """Dataset wrapper that assigns a fixed label to every image in a directory."""

    def __init__(self, directory, label, transform=None):
        self.directory = directory
        self.label = label
        self.transform = transform
        self.image_paths = sorted(
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith('.png')
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(self.label, dtype=torch.long)


def fine_tune(added_audio_dir="added_data", added_specs_dir="spectrograms_added",
              models_dir="models", model_path=None):
    """
    Fine-tunes a pre-trained base model to recognise a custom set of voices.

    Flow:
      1. Converts audio files in `added_audio_dir` into 4-second spectrograms (class_1).
      2. Applies image-level augmentation (×4) to the generated spectrograms so the
         tiny new class is better represented during training.
      3. Mixes the augmented class_1 spectrograms with a subset of the existing base
         class_0 spectrograms to prevent Catastrophic Forgetting.
      4. Applies class-weighted loss and WeightedRandomSampler to handle the remaining
         imbalance between class_0 and class_1.
      5. Fine-tunes the model with early stopping and saves the best weights.

    Args:
        added_audio_dir (str): Directory containing raw user audio. Defaults to "added_data".
        added_specs_dir (str): Directory to save generated spectrograms. Defaults to "spectrograms_added".
        models_dir (str): Directory to load/save model weights. Defaults to "models".
        model_path (str): Specific path to a pre-trained model. If None, auto-detects.
    """
    device = torch.device("cpu")

    if not os.path.exists(added_audio_dir) or not any(
        f.lower().endswith(SUPPORTED_AUDIO) for f in os.listdir(added_audio_dir)
    ):
        print("No supported audio files found in added_data/. Nothing to fine-tune.")
        return

    n_generated = preprocess_added_data(added_audio_dir, added_specs_dir)
    if n_generated == 0:
        print("No spectrograms generated. Aborting fine-tuning.")
        return

    specs_class1_dir = os.path.join(added_specs_dir, "class_1")
    augment_class1_specs(specs_class1_dir, factor=4)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    class1_dataset = FixedLabelDataset(specs_class1_dir, label=1, transform=transform)
    n_class1 = len(class1_dataset)

    class0_dir = "spectrograms/class_0"
    if os.path.exists(class0_dir) and os.listdir(class0_dir):
        class0_full = FixedLabelDataset(class0_dir, label=0, transform=transform)
        # Cap class_0 at 10× class_1 to preserve realistic imbalance without overwhelming new samples
        n_class0_keep = min(len(class0_full), n_class1 * 10)
        indices = random.sample(range(len(class0_full)), n_class0_keep)
        class0_subset = Subset(class0_full, indices)
        combined = ConcatDataset([class1_dataset, class0_subset])
        n_class0 = n_class0_keep
        print(f"Training mix — class_1: {n_class1} | class_0: {n_class0}")
    else:
        combined = class1_dataset
        n_class0 = 0
        print("Warning: spectrograms/class_0 not found. Training only on Class 1.")

    all_labels = [1] * n_class1 + [0] * n_class0
    class_counts = np.bincount(all_labels)
    sample_weights = [1.0 / class_counts[l] for l in all_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    weights_tensor = torch.tensor(1.0 / class_counts, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    train_loader = DataLoader(combined, batch_size=8, sampler=sampler)

    if model_path is None:
        candidates = [
            os.path.join(models_dir, "CustomCNN_fine_tuned.pth"),
            os.path.join(models_dir, "CustomCNN_best.pth"),
            os.path.join(models_dir, "DeepCNN_best.pth"),
            "CustomCNN_fine_tuned.pth",
            "CustomCNN_best.pth",
            "DeepCNN_best.pth",
        ]
        model_path = next((p for p in candidates if os.path.exists(p)), None)

    is_deepcnn = model_path and "DeepCNN" in model_path
    model = DeepCNN() if is_deepcnn else CustomCNN()
    save_name = "DeepCNN_fine_tuned.pth" if is_deepcnn else "CustomCNN_fine_tuned.pth"

    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print(f"Loaded weights from: {model_path}")
    else:
        print("No base model found — starting from scratch.")

    model.to(device)
    lr = 0.0005
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 30
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    save_path = os.path.join(models_dir, save_name)
    os.makedirs(models_dir, exist_ok=True)

    print(f"Fine-tuning for up to {epochs} epochs (lr={lr}, early stopping patience={patience})...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        if avg_loss < best_loss - 0.001:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    if not os.path.exists(save_path):
        torch.save(model.state_dict(), save_path)

    print(f"Fine-tuning complete. Model saved to: {save_path}")


if __name__ == "__main__":
    fine_tune()
