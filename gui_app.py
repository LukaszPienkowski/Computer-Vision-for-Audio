import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
import sounddevice as sd
import numpy as np
import torch
import os
import shutil
import threading
import time
from data_preprocess.audio_utils import generate_mel_spectrogram, spectrogram_to_image
from model import CustomCNN, DeepCNN

MODELS_DIR = "models"


class VoiceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Voice Recognition AI")
        self.geometry("620x780")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.is_recording = False
        self.recorded_audio = None
        self.sample_rate = 22050
        self.duration = 4.0

        # Queued files are tracked in memory only — start empty each session
        self.queued_files = []

        self.grid_columnconfigure(0, weight=1)

        # --- Header ---
        self.header = ctk.CTkLabel(self, text="Voice Classifier", font=("Roboto", 24, "bold"))
        self.header.grid(row=0, column=0, pady=20)

        # Classification Model selector
        self.class_model_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="transparent")
        self.class_model_frame.grid(row=1, column=0, pady=5)
        
        class_model_label = ctk.CTkLabel(self.class_model_frame, text="Model for Classification:", font=("Roboto", 12))
        class_model_label.pack(side="left", padx=10)

        self.class_model_var = ctk.StringVar(value=self._default_model_choice())
        self.class_model_dropdown = ctk.CTkOptionMenu(
            self.class_model_frame,
            variable=self.class_model_var,
            values=self._available_models(),
            command=self._on_classification_model_change,
            width=250
        )
        self.class_model_dropdown.pack(side="left", padx=10)

        self.status_label = ctk.CTkLabel(self, text="Ready to record (4s)", font=("Roboto", 14))
        self.status_label.grid(row=2, column=0, pady=10)

        self.record_button = ctk.CTkButton(
            self, text="Start Recording", command=self.toggle_recording,
            fg_color="#d32f2f", hover_color="#b71c1c", width=200, height=50
        )
        self.record_button.grid(row=3, column=0, pady=20)

        self.progress_bar = ctk.CTkProgressBar(self, width=400)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=4, column=0, pady=10)

        self.result_card = ctk.CTkFrame(self, corner_radius=15, fg_color="#2b2b2b")
        self.result_card.grid(row=5, column=0, pady=20, padx=40, sticky="nsew")
        self.result_card.grid_columnconfigure(0, weight=1)

        self.result_label = ctk.CTkLabel(
            self.result_card, text="Result: ---", font=("Roboto", 18, "bold")
        )
        self.result_label.grid(row=0, column=0, pady=20)

        # --- Fine-Tuning Section ---
        self.finetune_card = ctk.CTkFrame(self, corner_radius=15, fg_color="#1e2a38")
        self.finetune_card.grid(row=6, column=0, pady=10, padx=40, sticky="nsew")
        self.finetune_card.grid_columnconfigure(0, weight=1)

        ft_title = ctk.CTkLabel(
            self.finetune_card, text="Fine-Tuning", font=("Roboto", 16, "bold")
        )
        ft_title.grid(row=0, column=0, pady=(15, 5))

        # Model selector
        model_selector_label = ctk.CTkLabel(
            self.finetune_card, text="Base model:", font=("Roboto", 12)
        )
        model_selector_label.grid(row=1, column=0, pady=(4, 0))

        self.model_var = ctk.StringVar(value=self._default_model_choice())
        self.model_dropdown = ctk.CTkOptionMenu(
            self.finetune_card,
            variable=self.model_var,
            values=self._available_models(),
            width=300
        )
        self.model_dropdown.grid(row=2, column=0, pady=(2, 8))

        # File picker frame
        self.file_button_frame = ctk.CTkFrame(self.finetune_card, fg_color="transparent")
        self.file_button_frame.grid(row=3, column=0, pady=8)

        self.add_file_button = ctk.CTkButton(
            self.file_button_frame,
            text="Add Audio Files",
            command=self.add_wav_files,
            fg_color="#1565c0",
            hover_color="#0d47a1",
            width=150,
            height=40
        )
        self.add_file_button.grid(row=0, column=0, padx=5)

        self.clear_queue_button = ctk.CTkButton(
            self.file_button_frame,
            text="Clear Queue",
            command=self.clear_queue,
            fg_color="#d32f2f",
            hover_color="#b71c1c",
            width=100,
            height=40
        )
        self.clear_queue_button.grid(row=0, column=1, padx=5)

        self.added_files_label = ctk.CTkLabel(
            self.finetune_card,
            text="No files queued yet.",
            font=("Roboto", 12),
            text_color="#90caf9",
            wraplength=540
        )
        self.added_files_label.grid(row=4, column=0, pady=4)

        self.finetune_button = ctk.CTkButton(
            self.finetune_card,
            text="Run Fine-Tuning",
            command=self.run_fine_tuning,
            fg_color="#2e7d32",
            hover_color="#1b5e20",
            width=200,
            height=40
        )
        self.finetune_button.grid(row=5, column=0, pady=8)

        self.finetune_status_label = ctk.CTkLabel(
            self.finetune_card,
            text="",
            font=("Roboto", 12),
            wraplength=540
        )
        self.finetune_status_label.grid(row=6, column=0, pady=(4, 15))

        # Initialize model after UI variables are set
        self.model = self.load_model()

    # ------------------------------------------------------------------ #
    #  Model helpers
    # ------------------------------------------------------------------ #

    def _available_models(self):
        """Return list of .pth files found in the models/ folder."""
        if not os.path.exists(MODELS_DIR):
            return ["No models found"]
        models = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")])
        return models if models else ["No models found"]

    def _default_model_choice(self):
        """Prefer fine-tuned model, else fall back to first available."""
        models = self._available_models()
        for name in models:
            if "fine_tuned" in name:
                return name
        return models[0] if models else "No models found"

    def _on_classification_model_change(self, selected_name):
        """Callback for classification model dropdown."""
        if selected_name == "No models found":
            self.model = None
            return
        
        path = os.path.join(MODELS_DIR, selected_name)
        self.model = self.load_model(path=path)
        if self.model:
            self.status_label.configure(text=f"Loaded {selected_name} for classification.")

    def load_model(self, path=None):
        """Load a model from path. If path is None, use the current selection."""
        if path is None:
            selected_name = self.class_model_var.get()
            if selected_name == "No models found":
                # Fallback auto-detect
                candidates = [
                    os.path.join(MODELS_DIR, "CustomCNN_fine_tuned.pth"),
                    os.path.join(MODELS_DIR, "CustomCNN_best.pth"),
                    os.path.join(MODELS_DIR, "DeepCNN_best.pth"),
                    "CustomCNN_fine_tuned.pth",
                    "CustomCNN_best.pth",
                    "DeepCNN_best.pth"
                ]
                path = next((p for p in candidates if os.path.exists(p)), None)
            else:
                path = os.path.join(MODELS_DIR, selected_name)

        if path and os.path.exists(path):
            try:
                print(f"Loading model from: {path}")
                # Determine architecture based on filename
                if "DeepCNN" in path:
                    model = DeepCNN()
                else:
                    model = CustomCNN()
                    
                model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
                model.eval()
                return model
            except Exception as e:
                print(f"Error loading model {path}: {e}")
        else:
            print("No model files found.")
        return None

    # ------------------------------------------------------------------ #
    #  Recording
    # ------------------------------------------------------------------ #

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording_thread()

    def start_recording_thread(self):
        try:
            if not sd.query_devices(kind='input'):
                raise RuntimeError("No input device found.")
            self.is_recording = True
            self.record_button.configure(state="disabled", text="Recording...")
            self.result_label.configure(text="Listening...", text_color="white")
            self.status_label.configure(text="")
            self.progress_bar.set(0)
            threading.Thread(target=self.record_audio).start()
        except Exception as e:
            self.is_recording = False
            self.record_button.configure(state="normal", text="Start Recording")
            self.result_label.configure(text=f"Mic Error: {str(e)}", text_color="#ef5350")

    def record_audio(self):
        """
        Records audio from the user's default microphone for a specified duration,
        updates the UI progress bar during recording, and automatically triggers
        processing and classification once complete.
        """
        try:
            recording = sd.rec(
                int(self.duration * self.sample_rate),
                samplerate=self.sample_rate, channels=1
            )
            for i in range(101):
                if not self.is_recording:
                    break
                time.sleep(self.duration / 100)
                self.progress_bar.set(i / 100)
            sd.wait()
            self.recorded_audio = recording.flatten()
            self.is_recording = False
            self.after(0, self.process_and_classify)
        except Exception as e:
            self.is_recording = False
            self.after(0, lambda: self.record_button.configure(state="normal", text="Start Recording"))
            self.after(0, lambda: self.result_label.configure(
                text=f"Recording Error: {str(e)}", text_color="#ef5350"
            ))

    def process_and_classify(self):
        """
        Takes the raw audio recorded from the microphone, converts it into a 
        Mel spectrogram using the exact same normalization pipeline as the 
        training set, and feeds it into the loaded Neural Network for live inference.
        """
        self.record_button.configure(state="normal", text="Start Recording")
        self.status_label.configure(text="Processing audio...")

        if self.model is None:
            self.result_label.configure(text="Error: Model not found!", text_color="#ef5350")
            return

        try:
            S_norm = generate_mel_spectrogram(self.recorded_audio, self.sample_rate)
            img = spectrogram_to_image(S_norm, size=(128, 128))

            input_tensor = torch.from_numpy(np.array(img)).float().unsqueeze(0).unsqueeze(0) / 255.0
            input_tensor = (input_tensor - 0.5) / 0.5

            with torch.no_grad():
                output = self.model(input_tensor)
                prediction = torch.argmax(output, dim=1).item()
                confidence = torch.nn.functional.softmax(output, dim=1)[0][prediction].item()

            classes = ["Class 0", "Class 1"]
            result_text = f"Result: {classes[prediction]} ({confidence:.2%})"
            color = "#66bb6a" if prediction == 1 else "#ffa726"

            self.result_label.configure(text=result_text, text_color=color)
            self.status_label.configure(text="Classification complete.")

        except Exception as e:
            self.result_label.configure(text=f"Error: {str(e)}", text_color="#ef5350")

    # ------------------------------------------------------------------ #
    #  Fine-Tuning
    # ------------------------------------------------------------------ #

    def _refresh_queue_label(self):
        count = len(self.queued_files)
        if count == 0:
            self.added_files_label.configure(text="No files queued yet.", text_color="#90caf9")
        else:
            names = [os.path.basename(p) for p in self.queued_files]
            preview = ", ".join(names[:4]) + ("..." if count > 4 else "")
            self.added_files_label.configure(
                text=f"{count} file(s) queued: {preview}", text_color="#90caf9"
            )

    def clear_queue(self):
        """Clears the current fine-tuning queue."""
        self.queued_files.clear()
        self._refresh_queue_label()
        self.finetune_status_label.configure(
            text="Queue cleared.", text_color="#ffa726"
        )

    def add_wav_files(self):
        """Open a multi-file picker and queue selected audio files."""
        file_paths = filedialog.askopenfilenames(
            title="Select audio files for fine-tuning",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac"), ("All files", "*.*")]
        )
        if not file_paths:
            return

        added = 0
        for fp in file_paths:
            if fp not in self.queued_files:
                self.queued_files.append(fp)
                added += 1

        self._refresh_queue_label()
        self.finetune_status_label.configure(
            text=f"{added} file(s) added to queue.", text_color="#a5d6a7"
        )

    def run_fine_tuning(self):
        """
        Validates the queued audio files and triggers the backend fine-tuning 
        script in a separate background thread to prevent UI freezing. Once 
        complete, it automatically reloads the newly fine-tuned model weights.
        """
        if not self.queued_files:
            self.finetune_status_label.configure(
                text="No files queued. Add audio files first.", text_color="#ef5350"
            )
            return

        selected_model_name = self.model_var.get()
        if selected_model_name == "No models found":
            self.finetune_status_label.configure(
                text="No base model available in models/ folder.", text_color="#ef5350"
            )
            return

        model_path = os.path.join(MODELS_DIR, selected_model_name)

        self.finetune_button.configure(state="disabled", text="Fine-Tuning...")
        self.add_file_button.configure(state="disabled")
        self.clear_queue_button.configure(state="disabled")
        self.model_dropdown.configure(state="disabled")
        self.finetune_status_label.configure(text="Copying files & starting fine-tuning...", text_color="white")

        # Clear and repopulate added_data/ so stale files from previous
        # fine-tuning runs don't contaminate the new training set.
        added_dir = "added_data"
        if os.path.exists(added_dir):
            shutil.rmtree(added_dir)
        os.makedirs(added_dir)
        for fp in self.queued_files:
            dest = os.path.join(added_dir, os.path.basename(fp))
            try:
                shutil.copy2(fp, dest)
            except Exception as e:
                print(f"Warning: could not copy {fp}: {e}")

        threading.Thread(
            target=self._fine_tune_thread,
            args=(model_path,),
            daemon=True
        ).start()

    def _fine_tune_thread(self, model_path):
        try:
            from fine_tune_model import fine_tune
            fine_tune(model_path=model_path)
            self.after(0, self._on_fine_tune_done)
        except Exception as e:
            self.after(0, lambda err=str(e): self._on_fine_tune_error(err))

    def _on_fine_tune_done(self):
        # Determine which fine-tuned model was produced based on the selected base
        selected = self.model_var.get()
        if "DeepCNN" in selected:
            fine_tuned_name = "DeepCNN_fine_tuned.pth"
        else:
            fine_tuned_name = "CustomCNN_fine_tuned.pth"

        new_path = os.path.join(MODELS_DIR, fine_tuned_name)
        self.model = self.load_model(path=new_path if os.path.exists(new_path) else None)

        # Refresh the model dropdowns
        updated_models = self._available_models()
        self.model_dropdown.configure(values=updated_models)
        self.class_model_dropdown.configure(values=updated_models)

        if fine_tuned_name in updated_models:
            self.model_var.set(fine_tuned_name)
            self.class_model_var.set(fine_tuned_name)
            self._on_classification_model_change(fine_tuned_name)
        else:
            self.model_var.set(self._default_model_choice())
            self.class_model_var.set(self._default_model_choice())
            self._on_classification_model_change(self.class_model_var.get())

        # Clear the queue
        self.queued_files = []
        self._refresh_queue_label()

        self.finetune_status_label.configure(
            text="Fine-tuning complete! Model reloaded.", text_color="#a5d6a7"
        )
        self.finetune_button.configure(state="normal", text="Run Fine-Tuning")
        self.add_file_button.configure(state="normal")
        self.clear_queue_button.configure(state="normal")
        self.model_dropdown.configure(state="normal")

    def _on_fine_tune_error(self, err):
        self.finetune_status_label.configure(
            text=f"Fine-tuning failed: {err}", text_color="#ef5350"
        )
        self.finetune_button.configure(state="normal", text="Run Fine-Tuning")
        self.add_file_button.configure(state="normal")
        self.clear_queue_button.configure(state="normal")
        self.model_dropdown.configure(state="normal")


if __name__ == "__main__":
    app = VoiceApp()
    app.mainloop()
