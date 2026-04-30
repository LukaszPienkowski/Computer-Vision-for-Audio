import os
import random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# ─────────────────────────────────────────────
#  Device
# ─────────────────────────────────────────────
device = torch.device("cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────
#  Shared image transform
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


# ─────────────────────────────────────────────
#  Speaker-level dataset splitting
# ─────────────────────────────────────────────
def speaker_level_split(dataset, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Splits an ImageFolder dataset into train / val / test subsets at the
    **speaker** level so that all spectrogram segments from a single speaker
    end up entirely in one partition.  This prevents data leakage where
    segments from the same recording appear in both train and test.

    Filenames are expected to follow the convention ``{speaker_id}_{index}.png``
    (produced by ``generating_spectrograms.py``).  The speaker_id prefix is
    extracted by splitting on the first underscore.

    Each class's speakers are shuffled independently and then partitioned into
    train / val / test so that the class ratio is preserved across splits.

    Args:
        dataset (ImageFolder): The full spectrogram dataset.
        val_ratio (float):  Fraction of speakers to allocate to validation (default: 0.15).
        test_ratio (float): Fraction of speakers to allocate to test (default: 0.15).
        seed (int):         Random seed for reproducibility (default: 42).

    Returns:
        tuple[Subset, Subset, Subset]: ``(train_subset, val_subset, test_subset)``
    """
    random.seed(seed)

    # Group sample indices by (speaker_id, class_label) key
    speaker_to_indices = defaultdict(list)
    for idx, (path, label) in enumerate(dataset.samples):
        filename = os.path.basename(path)
        speaker_id = filename.split("_")[0]
        speaker_to_indices[(speaker_id, label)].append(idx)

    # Separate speakers by class so each class is split proportionally
    class_0_speakers = [k for k in speaker_to_indices if k[1] == 0]
    class_1_speakers = [k for k in speaker_to_indices if k[1] == 1]
    random.shuffle(class_0_speakers)
    random.shuffle(class_1_speakers)

    def _split(speakers):
        """Partition a speaker list into (train, val, test) slices."""
        n = len(speakers)
        n_test = max(1, int(n * test_ratio))
        n_val  = max(1, int(n * val_ratio))
        n_train = n - n_val - n_test
        return (
            speakers[:n_train],
            speakers[n_train: n_train + n_val],
            speakers[n_train + n_val:],
        )

    train_0, val_0, test_0 = _split(class_0_speakers)
    train_1, val_1, test_1 = _split(class_1_speakers)

    def _collect(keys):
        idxs = []
        for k in keys:
            idxs.extend(speaker_to_indices[k])
        return idxs

    train_indices = _collect(train_0 + train_1)
    val_indices   = _collect(val_0   + val_1)
    test_indices  = _collect(test_0  + test_1)

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


def make_weighted_loader(subset, batch_size=32):
    """
    Builds a DataLoader with a WeightedRandomSampler that rebalances batches
    by inverse class frequency, so the minority class_1 is seen as often as
    class_0 during training.

    Args:
        subset (Subset):  A dataset subset produced by ``speaker_level_split``.
        batch_size (int): Mini-batch size (default: 32).

    Returns:
        DataLoader: A DataLoader with weighted sampling applied.
    """
    labels = [subset.dataset.targets[i] for i in subset.indices]
    counts = np.bincount(labels)
    sample_weights = [1.0 / counts[l] for l in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return DataLoader(subset, batch_size=batch_size, sampler=sampler)


# ─────────────────────────────────────────────
#  Module-level dataset loading
# ─────────────────────────────────────────────
dataset_path = "spectrograms"
full_dataset = None
train_ds = val_ds = test_ds = None
train_loader = val_loader = test_loader = None

if os.path.exists(dataset_path):
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    train_ds, val_ds, test_ds = speaker_level_split(full_dataset)

    train_labels   = [full_dataset.targets[i] for i in train_ds.indices]
    class_counts_  = np.bincount(train_labels)

    train_loader = make_weighted_loader(train_ds, batch_size=32)
    val_loader   = DataLoader(val_ds,  batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

    print(f"Dataset loaded : {len(full_dataset)} images total")
    print(f"Split          — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"Train class counts — class_0: {class_counts_[0]} | class_1: {class_counts_[1]}")
    print(f"Loss weights       — class_0: {1.0/class_counts_[0]:.4f} | class_1: {1.0/class_counts_[1]:.4f}")
else:
    print("Spectrograms directory not found! Run main.py first.")


# ─────────────────────────────────────────────
#  Model architectures
# ─────────────────────────────────────────────
class CustomCNN(nn.Module):
    """
    A lightweight Convolutional Neural Network for audio spectrogram
    classification.  Three convolutional layers with batch normalisation
    and max pooling, followed by a fully connected classifier with dropout.

    Architecture:
        Conv(1→32) → BN → ReLU → MaxPool
        Conv(32→64) → ReLU → MaxPool
        Conv(64→128) → BN → ReLU → AdaptiveAvgPool(4×4)
        Flatten → Linear(2048→128) → ReLU → Dropout(0.4) → Linear(128→2)
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class DeepCNN(nn.Module):
    """
    A deeper Convolutional Neural Network for audio spectrogram classification.
    Five convolutional layers (arranged in two VGG-style blocks) with batch
    normalisation, followed by a larger fully connected head with higher
    dropout to prevent overfitting.

    Architecture:
        Conv(1→32) → BN → ReLU → Conv(32→32) → BN → ReLU → MaxPool
        Conv(32→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool
        Conv(64→128) → BN → ReLU → AdaptiveAvgPool(4×4)
        Flatten → Linear(2048→256) → ReLU → Dropout(0.5) → Linear(256→2)
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────
def evaluate_detailed(model, dataloader):
    """
    Evaluates a PyTorch model against a given DataLoader and computes a full
    set of binary-classification performance metrics.

    Metrics returned
    ----------------
    - **Accuracy**  : fraction of correct predictions overall.
    - **Precision** : weighted-average precision across both classes.
    - **Recall**    : weighted-average recall across both classes.
    - **F1-Score**  : weighted-average F1 across both classes.
    - **FAR** (False Acceptance Ratio) : ``FP / (FP + TN)`` — fraction of
      class_0 samples incorrectly admitted as class_1.
    - **FRR** (False Rejection Ratio) : ``FN / (FN + TP)`` — fraction of
      class_1 samples incorrectly rejected as class_0.

    Args:
        model (nn.Module): The trained PyTorch model to evaluate.
        dataloader (DataLoader): DataLoader for the evaluation split.

    Returns:
        dict: Keys — ``Accuracy``, ``Precision``, ``Recall``, ``F1-Score``,
              ``FAR``, ``FRR``.
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    # Confusion matrix — rows: actual, cols: predicted; labels=[0, 1]
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # FAR: class_0 incorrectly accepted (predicted as class_1)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    # FRR: class_1 incorrectly rejected (predicted as class_0)
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "FAR": far,
        "FRR": frr,
    }


# ─────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────
def train_model(model, name, epochs=20, optimizer_name="Adam"):
    """
    Trains a model using class-weighted loss and early stopping on the
    **validation** set.  Saves the best weights to ``models/``.

    Two optimisers are supported for ablation:

    * **Adam** (default) — ``lr=0.001``, no scheduler.
    * **SGD**            — ``lr=0.01``, momentum=0.9, weight_decay=1e-4,
      StepLR scheduler (step_size=5, gamma=0.5).

    The training loss and validation loss are recorded per epoch and saved
    as a PNG plot in ``plots/``.  FAR and FRR are printed for train, val, and
    test splits after training is complete.

    Args:
        model (nn.Module):      The PyTorch model to train.
        name (str):             Architecture identifier (e.g. ``"CustomCNN"``).
        epochs (int):           Maximum number of training epochs (default: 20).
        optimizer_name (str):   ``"Adam"`` or ``"SGD"`` (default: ``"Adam"``).

    Returns:
        dict: Evaluation metrics on the held-out **test** set.
    """
    print(f"\n{'='*55}")
    print(f"  Training {name}  |  Optimizer: {optimizer_name}")
    print(f"{'='*55}")
    model = model.to(device)

    # Class-weighted loss (inverse frequency)
    t_labels = [full_dataset.targets[i] for i in train_ds.indices]
    t_counts = np.bincount(t_labels)
    loss_weights = torch.tensor(1.0 / t_counts, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    if optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    else:  # Adam
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = None

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    # Save path — Adam run uses the canonical name (backward-compatible with gui_app.py)
    save_name = (
        f"{name}_best.pth" if optimizer_name == "Adam" else f"{name}_SGD_best.pth"
    )
    save_path = os.path.join("models", save_name)
    os.makedirs("models", exist_ok=True)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # ── Training pass ──
        model.train()
        running_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if scheduler:
            scheduler.step()
        avg_train = running_loss / len(train_loader)
        train_losses.append(avg_train)

        # ── Validation pass ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                val_loss += criterion(model(imgs), lbls).item()
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)

        print(
            f"  Epoch {epoch+1:>3}/{epochs} — "
            f"Train Loss: {avg_train:.4f}  |  Val Loss: {avg_val:.4f}"
        )

        # ── Early stopping (on val) ──
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("  Early stopping triggered.")
                break

    # ── Save loss curve ──
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.title(f"{name} ({optimizer_name}) — Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{name}_{optimizer_name}_loss.png", dpi=100)
    plt.close()

    # ── Load best weights for final evaluation ──
    model.load_state_dict(torch.load(save_path, map_location=device))

    train_m = evaluate_detailed(model, train_loader)
    val_m   = evaluate_detailed(model, val_loader)
    test_m  = evaluate_detailed(model, test_loader)

    print(f"\n  {'Split':<8} {'Acc':>6} {'F1':>6} {'FAR':>6} {'FRR':>6}")
    print(f"  {'-'*36}")
    for split_name, m in [("Train", train_m), ("Val", val_m), ("Test", test_m)]:
        print(
            f"  {split_name:<8} "
            f"{m['Accuracy']:>6.3f} "
            f"{m['F1-Score']:>6.3f} "
            f"{m['FAR']:>6.3f} "
            f"{m['FRR']:>6.3f}"
        )

    return test_m


# ─────────────────────────────────────────────
#  Standalone execution — train all variants & compare
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if full_dataset is None:
        print("No spectrogram dataset found. Run main.py first.")
    else:
        results = {}

        # ── Technique (b): different architectures  ──
        # ── Technique (c): Adam vs SGD optimisers   ──
        for arch_name, arch_cls in [("CustomCNN", CustomCNN), ("DeepCNN", DeepCNN)]:
            for opt in ["Adam", "SGD"]:
                key = f"{arch_name} ({opt})"
                results[key] = train_model(arch_cls(), arch_name, optimizer_name=opt)

        # ── Comparative table ──
        df = pd.DataFrame(results).T
        print("\n" + "=" * 60)
        print("  COMPARATIVE RESULTS — ALL MODELS (Test Set)")
        print("=" * 60)
        print(df.to_string(float_format=lambda x: f"{x:.4f}"))

        os.makedirs("plots", exist_ok=True)
        df.to_csv("plots/model_comparison.csv")
        print("\nComparison table saved → plots/model_comparison.csv")

        # ── Bar chart ──
        metrics_to_plot = ["Accuracy", "F1-Score", "FAR", "FRR"]
        df[metrics_to_plot].plot(kind="bar", figsize=(14, 6))
        plt.title("Model Comparison — Test Set Metrics (lower FAR/FRR is better)")
        plt.ylabel("Score")
        plt.ylim(0, 1.15)
        plt.xticks(rotation=15, ha="right")
        plt.legend(loc="upper right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/model_comparison.png", dpi=120)
        plt.show()
        print("Comparison chart saved → plots/model_comparison.png")