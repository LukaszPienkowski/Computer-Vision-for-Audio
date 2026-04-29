import os
import random
from PIL import Image
from audio_utils import time_mask, freq_mask, add_noise_to_image

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
    
    for f in files:
        file_path = os.path.join(added_dir, f)
        try:
            img = Image.open(file_path)
            base_name = f.rsplit('.', 1)[0]
            
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
    Balances the spectrogram dataset by augmenting Class 1 to match the count of Class 0.

    Counts PNG images in `spectrograms/class_0` and `spectrograms/class_1`. If Class 1
    has fewer images, generates the required number of augmented copies using randomly
    chosen image-level transformations (time mask, frequency mask, Gaussian noise),
    sampling randomly from the existing original Class 1 spectrograms.
    """
    class_0_dir = "spectrograms/class_0"
    class_1_dir = "spectrograms/class_1"
    
    if not os.path.exists(class_0_dir) or not os.path.exists(class_1_dir):
        print("Spectrogram directories not found. Run generating_spectrograms.py first.")
        return
        
    class_0_files = [f for f in os.listdir(class_0_dir) if f.endswith('.png')]
    class_1_files = [f for f in os.listdir(class_1_dir) if f.endswith('.png') and not '_aug_' in f]
    
    target_count = len(class_0_files)
    current_count = len([f for f in os.listdir(class_1_dir) if f.endswith('.png')])
    needed_count = target_count - current_count
    
    print(f"Class 0 count: {target_count}")
    print(f"Class 1 count: {current_count}")
    
    if needed_count <= 0:
        print("No augmentation needed. Classes are already balanced.")
        return
        
    print(f"Generating {needed_count} augmented files for Class 1...")
    
    augmentations = [
        ("tmask", lambda img: time_mask(img)),
        ("fmask", lambda img: freq_mask(img)),
        ("noise", lambda img: add_noise_to_image(img, factor=random.uniform(0.01, 0.05)))
    ]
    
    for i in range(needed_count):
        file_to_augment = random.choice(class_1_files)
        file_path = os.path.join(class_1_dir, file_to_augment)
        
        try:
            img = Image.open(file_path)
            
            aug_name, aug_func = random.choice(augmentations)
            aug_img = aug_func(img)
            
            base_name = file_to_augment.replace('.png', '')
            new_file_name = f"{base_name}_aug_{aug_name}_{i}.png"
            new_file_path = os.path.join(class_1_dir, new_file_name)
            
            aug_img.save(new_file_path)
            
            if (i+1) % 50 == 0:
                print(f"Generated {i+1}/{needed_count} files...")
                
        except Exception as e:
            print(f"Error augmenting {file_to_augment}: {e}")
            
    print(f"Successfully balanced classes. Class 1 now has {target_count} files.")

if __name__ == "__main__":
    main()
