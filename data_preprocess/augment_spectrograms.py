import os
import random
from PIL import Image
from .audio_utils import time_mask, freq_mask, add_noise_to_image

def augment_added_data(added_dir="spectrograms_added/class_1", factor=2):
    """
    Applies image-level augmentations to spectrogram PNG files in a given directory.

    For each original (non-augmented) image found, generates `factor` additional
    augmented copies using randomly chosen transformations: time masking, frequency
    masking, or Gaussian noise. Augmented files are named with an `_aug_` infix to
    distinguish them from original images and prevent re-augmentation.

    Args:
        added_dir (str): Path to the directory containing spectrogram PNG files to augment
                         (default: 'spectrograms_added/class_1').
        factor (int): Number of augmented copies to generate per original image (default: 2).
    """
    if not os.path.exists(added_dir):
        return
        
    files = [f for f in os.listdir(added_dir) if f.endswith('.png') and not '_aug_' in f]
    
    if not files:
        return
        
    print(f"Applying image augmentation to {len(files)} files in {added_dir}...")
    
    augmentations = [
        ("tmask", lambda img: time_mask(img)),
        ("fmask", lambda img: freq_mask(img)),
        ("noise", lambda img: add_noise_to_image(img, factor=random.uniform(0.01, 0.05)))
    ]
    
    all_files = set(os.listdir(added_dir))
    
    for f in files:
        base_name = f.rsplit('.', 1)[0]
        
        # Check if this specific base_name has already been augmented
        already_augmented = any(
            existing_f.startswith(f"{base_name}_aug_") for existing_f in all_files
        )
        if already_augmented:
            continue
            
        file_path = os.path.join(added_dir, f)
        try:
            img = Image.open(file_path)
            
            for i in range(factor):
                aug_name, aug_func = random.choice(augmentations)
                aug_img = aug_func(img)
                
                new_file_name = f"{base_name}_aug_{aug_name}_{i}.png"
                new_file_path = os.path.join(added_dir, new_file_name)
                aug_img.save(new_file_path)
                
        except Exception as e:
            print(f"Error augmenting added data file {f}: {e}")

def main():
    """
    Applies 2× image-level augmentation to every original spectrogram in class_1.

    The dataset is intentionally imbalanced (class_0 >> class_1). Rather than
    forcing a 1:1 ratio, this function only augments class_1 to improve its
    representation without destroying the natural imbalance that reflects the
    real task: detecting a small set of target voices among many others.
    """
    class_0_dir = "spectrograms/class_0"
    class_1_dir = "spectrograms/class_1"
    
    if not os.path.exists(class_0_dir) or not os.path.exists(class_1_dir):
        print("Spectrogram directories not found. Run generating_spectrograms.py first.")
        return
        
    class_0_files = [f for f in os.listdir(class_0_dir) if f.endswith('.png')]
    class_1_files = [f for f in os.listdir(class_1_dir) if f.endswith('.png') and not '_aug_' in f]
    
    print(f"Class 0: {len(class_0_files)} | Class 1: {len(class_1_files)}")
    augment_added_data(added_dir=class_1_dir, factor=2)
    final_class_1_count = len([f for f in os.listdir(class_1_dir) if f.endswith('.png')])
    print(f"Augmentation complete. Class 1 now has {final_class_1_count} files.")


if __name__ == "__main__":
    main()
