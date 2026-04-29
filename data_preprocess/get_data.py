import os
import io
import soundfile as sf
import shutil
from datasets import load_dataset, Audio
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
    
def main(subset="en", limit=300, dataset_name="facebook/voxpopuli"):
    """
    Downloads audio data from a specified Hugging Face dataset, filters it, 
    and organizes it into two classes to create a local dataset.
    
    Args:
        subset (str): The subset/language of the dataset to download (default: "en").
        limit (int): The maximum number of audio files to download (default: 300).
        dataset_name (str): The name of the dataset on Hugging Face (default: "facebook/voxpopuli").
        
    Details:
        - Streams data to avoid downloading the entire dataset at once.
        - Ensures only one recording is kept per speaker.
        - Filters out audio recordings shorter than 25 seconds.
        - Automatically splits data into 'class_1' (15%) and 'class_0' (85%).
        - Cleans and recreates local 'data/class_0' and 'data/class_1' directories before saving.
        - Automatically collects any local .wav recordings from the 'my_records' folder 
          and appends them to 'class_0' to mitigate microphone domain shift.
    """
    ds = load_dataset(path=dataset_name, name=subset, split="train", streaming=True, token=hf_token)
    ds = ds.cast_column("audio", Audio(decode=False))
    
    for label in ["class_0", "class_1"]:
        dir_path = f"data/{label}"
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)
    
    downloaded_count = 0
    seen_speakers = set()
    
    print(f"Starting download of {limit} files (15% class_1, 85% class_0)...")
    
    for example in ds:
        if downloaded_count >= limit:
            break
            
        speaker_id = example['speaker_id']
        if speaker_id in seen_speakers:
            continue
            
        audio_bytes = example['audio']['bytes']
        data, samplerate = sf.read(io.BytesIO(audio_bytes))
        duration = len(data) / samplerate
        
        if duration <= 25.0:
            continue
            
        label = "class_1" if downloaded_count < int(limit * 0.15) else "class_0" 
        
        path = f"data/{label}/{speaker_id}.wav"
        sf.write(path, data, samplerate)
        seen_speakers.add(speaker_id)
        downloaded_count += 1
        
        if downloaded_count % 50 == 0:
            print(f"Downloaded {downloaded_count}/{limit} files...")
        
    print(f"Successfully finished downloading {downloaded_count} files from Hugging Face.")
    
    # Add local recordings to class_0 (non-speaking background noise)
    my_records_dir = "my_records"
    if os.path.exists(my_records_dir):
        print(f"Checking for local non-speaking recordings in '{my_records_dir}'...")
        local_count = 0
        for filename in os.listdir(my_records_dir):
            if filename.endswith(".wav"):
                src_path = os.path.join(my_records_dir, filename)
                dest_path = os.path.join("data", "class_0", f"local_record_{local_count}.wav")
                shutil.copy2(src_path, dest_path)
                local_count += 1
        if local_count > 0:
            print(f"Added {local_count} local recordings to class_0.")
        else:
            print(f"No .wav files found in {my_records_dir}.")
    else:
        print(f"'{my_records_dir}' folder not found. Skipping local data integration.")
        
    os._exit(0)

if __name__ == "__main__":
    main()