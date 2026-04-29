import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Force CPU
device = torch.device("cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset_path = 'spectrograms'
if not os.path.exists(dataset_path):
    print("Spectrograms directory not found!")
else:
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_ds, test_ds = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    print(f"Dataset loaded: {len(full_dataset)} images.")

class CustomCNN(nn.Module):
    """
    A lightweight Convolutional Neural Network architecture designed for classifying 
    audio spectrogram images. It features three convolutional layers with batch 
    normalization and max pooling, followed by a fully connected classifier.
    """
    def __init__(self, num_classes=2):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

class DeepCNN(nn.Module):
    """
    A deeper, more complex Convolutional Neural Network architecture for audio spectrogram 
    classification. Features five convolutional layers with batch normalization and max 
    pooling, followed by a larger fully connected classifier with higher dropout to 
    prevent overfitting on complex datasets.
    """
    def __init__(self, num_classes=2):
        super(DeepCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

def evaluate_detailed(model, dataloader):
    """
    Evaluates a PyTorch model against a given dataloader and calculates detailed performance metrics.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        dataloader (DataLoader): The DataLoader containing the evaluation dataset.

    Returns:
        dict: A dictionary containing 'Accuracy', 'Precision', 'Recall', and 'F1-Score'.
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}

def train_model(model, name, epochs=50):
    """
    Trains a given PyTorch model, evaluates its performance on the test set, and saves 
    the resulting weights to the 'models/' directory.

    Args:
        model (nn.Module): The PyTorch model to train.
        name (str): The name identifier for the model (used for saving the weights file).
        epochs (int, optional): The maximum number of training epochs. Defaults to 50.

    Returns:
        dict: The evaluation metrics for the trained model on the test set.
    """
    print(f"Training {name}...")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(test_loader)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        
        # Early Stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/{name}_best.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
                
    # Load best weights for final evaluation
    model.load_state_dict(torch.load(f"models/{name}_best.pth", map_location=device))
    metrics = evaluate_detailed(model, test_loader)
    return metrics

if __name__ == "__main__":
    results = {}
    results["CustomCNN"] = train_model(CustomCNN(), "CustomCNN")
    results["DeepCNN"] = train_model(DeepCNN(), "DeepCNN")
    
    # Table Representation
    df_results = pd.DataFrame(results).T
    print("\n--- Comparative Table ---")
    print(df_results)
    
    # Graphical Representation
    df_results.plot(kind='bar', figsize=(12, 6))
    plt.title("Comparison of Model Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.show()