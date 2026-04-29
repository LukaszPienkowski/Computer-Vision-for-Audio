import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from data_preprocess.audio_utils import generate_mel_spectrogram, save_spectrogram
from model import CustomCNN, DeepCNN

# 2. Preprocessing logic for added data
def preprocess_added_data(input_base, output_base):
    if not os.path.exists(input_base):
        print(f"Directory {input_base} does not exist.")
        return
        
    class1_dir = os.path.join(output_base, "class_1")
    os.makedirs(class1_dir, exist_ok=True)
    
    import librosa
    
    for filename in os.listdir(input_base):
        if not filename.endswith(".wav"):
            continue
            
        audio_path = os.path.join(input_base, filename)
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Use 4-second chunking like the main pipeline
            chunk_length = 4 * sr
            base_name = os.path.splitext(filename)[0]
            
            # Split the audio into 4-second chunks
            for i in range(0, len(y) - chunk_length, chunk_length):
                y_chunk = y[i:i + chunk_length]
                S_dB_norm = generate_mel_spectrogram(y_chunk, sr)
                out_name = f"{base_name}_{i}.png"
                out_path = os.path.join(class1_dir, out_name)
                save_spectrogram(S_dB_norm, out_path)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Fixed dataset that overrides folder names to force label = 1
class FixedLabelDataset(Dataset):
    def __init__(self, directory, label, transform=None):
        self.directory = directory
        self.label = label
        self.transform = transform
        self.image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(self.label, dtype=torch.long)

def fine_tune(added_audio_dir="added_data", added_specs_dir="spectrograms_added", models_dir="models", model_path=None):
    device = torch.device("cpu")
    
    if not os.path.exists(added_audio_dir) or not os.listdir(added_audio_dir):
        print("No added audio data to fine-tune on.")
        return

    preprocess_added_data(added_audio_dir, added_specs_dir)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Use FixedLabelDataset to guarantee all added images are labelled as Class 1
    specs_class1_dir = os.path.join(added_specs_dir, "class_1")
    if not os.path.exists(specs_class1_dir) or not os.listdir(specs_class1_dir):
        print("No spectrogram images found after preprocessing.")
        return

    try:
        added_dataset = FixedLabelDataset(specs_class1_dir, label=1, transform=transform)
        
        # Mix in Class 0 data to prevent catastrophic forgetting
        class0_dir = "spectrograms/class_0"
        if os.path.exists(class0_dir):
            from torch.utils.data import ConcatDataset, Subset
            import random
            class0_dataset = FixedLabelDataset(class0_dir, label=0, transform=transform)
            num_added = len(added_dataset)
            if len(class0_dataset) > num_added:
                indices = random.sample(range(len(class0_dataset)), num_added)
                class0_subset = Subset(class0_dataset, indices)
            else:
                class0_subset = class0_dataset
            
            combined_dataset = ConcatDataset([added_dataset, class0_subset])
            train_loader = DataLoader(combined_dataset, batch_size=8, shuffle=True)
            print(f"Mixed {num_added} new Class 1 samples with {len(class0_subset)} Class 0 samples.")
        else:
            train_loader = DataLoader(added_dataset, batch_size=8, shuffle=True)
            print("Warning: spectrograms/class_0 not found. Training only on Class 1.")
            
    except Exception as e:
        print(f"No valid data found to train on: {e}")
        return

    # Resolve model to load: explicit path > fine-tuned in models/ > best in models/ > root fallbacks
    if model_path is None:
        candidates = [
            os.path.join(models_dir, "CustomCNN_fine_tuned.pth"),
            os.path.join(models_dir, "CustomCNN_best.pth"),
            os.path.join(models_dir, "DeepCNN_best.pth"),
            "CustomCNN_fine_tuned.pth",
            "CustomCNN_best.pth",
            "DeepCNN_best.pth"
        ]
        model_path = next((p for p in candidates if os.path.exists(p)), None)

    # Determine architecture
    is_deepcnn = model_path and "DeepCNN" in model_path
    if is_deepcnn:
        model = DeepCNN()
        save_name = "DeepCNN_fine_tuned.pth"
    else:
        model = CustomCNN()
        save_name = "CustomCNN_fine_tuned.pth"

    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print(f"Loaded model weights from: {model_path}")
    else:
        print("Starting from scratch (no base model found).")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # DeepCNN has more parameters and deeper layers, so it needs a higher learning rate 
    # to adapt quickly during fine-tuning on a small dataset.
    lr = 0.001 if is_deepcnn else 0.0005
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    epochs = 30
    print(f"Fine-tuning for {epochs} epochs with lr={lr}...")
    model.train()
    
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    save_path = os.path.join(models_dir, save_name)
    
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss/len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        # Early Stopping on train loss for fine-tuning
        if avg_loss < best_loss - 0.001:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered during fine-tuning!")
                break
    
    if not os.path.exists(save_path):
        torch.save(model.state_dict(), save_path)
        
    print(f"Fine-tuning complete. Saved as {save_path}")

if __name__ == "__main__":
    fine_tune()
