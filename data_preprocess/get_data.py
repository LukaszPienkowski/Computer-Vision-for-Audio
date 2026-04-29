import os
import io
import random
import soundfile as sf
import shutil
from datasets import load_dataset, Audio
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")


def main(limit=300, class1_speakers=5, dataset_name="facebook/voxpopuli"):
    """
    Downloads English audio data from VoxPopuli, collects 'limit' unique speakers
    into class_0, then randomly moves 'class1_speakers' of them into class_1.

    This ensures both classes are from the same domain (continuous English speech),
    forcing the model to learn actual voice characteristics rather than audio format
    differences. Class 1 is intentionally small, Class 0 is intentionally huge.

    Args:
        limit (int): Number of unique speakers to download into class_0 (default: 500).
        class1_speakers (int): How many speakers to randomly promote to class_1 (default: 7).
        dataset_name (str): HuggingFace dataset name (default: 'facebook/voxpopuli').

    Details:
        - Streams data to avoid downloading the entire dataset at once.
        - Ensures only one recording is kept per speaker.
        - Filters out recordings shorter than 25 seconds.
        - After collecting all class_0 speakers, randomly selects class1_speakers
          of them and moves their audio files into class_1.
        - Automatically copies any .wav files found in 'my_records/' into
          class_0 to inject local microphone noise for domain-shift mitigation.
    """
    print("Setting up dataset directories...")
    for label in ["class_0", "class_1"]:
        dir_path = f"data/{label}"
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)

    ds = load_dataset(path=dataset_name, name="en", split="train", streaming=True, token=hf_token)
    ds = ds.cast_column("audio", Audio(decode=False))

    downloaded = 0
    seen_speakers = set()
    # Track which file belongs to which speaker so we can move them later
    speaker_to_file = {}

    print(f"Streaming VoxPopuli (English) — collecting {limit} unique speakers into class_0...")

    for example in ds:
        if downloaded >= limit:
            break

        speaker_id = example['speaker_id']
        if speaker_id in seen_speakers:
            continue

        audio_bytes = example['audio']['bytes']
        data, samplerate = sf.read(io.BytesIO(audio_bytes))
        duration = len(data) / samplerate

        if duration <= 25.0:
            continue

        seen_speakers.add(speaker_id)
        file_path = f"data/class_0/{speaker_id}.wav"
        sf.write(file_path, data, samplerate)
        speaker_to_file[speaker_id] = file_path
        downloaded += 1

        if downloaded % 10 == 0:
            print(f"  Downloaded {downloaded}/{limit} speakers...")

    print(f"Collected {downloaded} speakers into class_0.")

    # Randomly promote a small subset to class_1
    n_to_move = random.randint(5, class1_speakers)
    target_speakers = random.sample(list(speaker_to_file.keys()), min(n_to_move, len(speaker_to_file)))
    print(f"Randomly promoting {len(target_speakers)} speakers to class_1: {target_speakers}")

    for speaker_id in target_speakers:
        src = speaker_to_file[speaker_id]
        dst = f"data/class_1/{speaker_id}.wav"
        shutil.move(src, dst)

    print(f"Final split — Class 0: {downloaded - len(target_speakers)} | Class 1: {len(target_speakers)}")

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

    print("Data ingestion complete!")


if __name__ == "__main__":
    main()