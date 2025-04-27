import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import os
import json
from audio_recorder import AudioRecorder
from command_executor import CommandExecutor
from command_organizer import CommandOrganizer
import torch
import librosa
import threading
from pathlib import Path

class VoiceCommandApp:
    def __init__(self, root, model_trainer, command_organizer):
        self.root = root
        self.root.title("Voice Command System")
        self.root.geometry("1000x700")
        
        self.model_trainer = model_trainer
        self.command_organizer = command_organizer
        self.audio_recorder = AudioRecorder()
        self.command_executor = CommandExecutor()
        
        # Setup GUI first so we have access to background operations text
        self.setup_gui()
        
        # Initialize model after GUI is setup
        self.initialize_model()
        
        # Update recordings list
        self.update_recordings_list()

    def initialize_model(self):
        """Initialize the model either by loading trained model or base model"""
        try:
            self.add_background_operation("Initializing model...")
            if not self.model_trainer.load_trained_model():
                self.add_background_operation("No trained model found, loading base model...")
                if not self.model_trainer.load_base_model():
                    self.add_background_operation("Error: Failed to load base model")
                else:
                    self.add_background_operation("Base model loaded successfully")
            else:
                self.add_background_operation("Trained model loaded successfully")
        except Exception as e:
            self.add_background_operation(f"Error initializing model: {str(e)}")

    def setup_gui(self):
        # Create main frame with three columns
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left panel for recording controls
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Record button
        self.record_button = ttk.Button(
            left_panel,
            text="Press and Hold to Record",
            style='Record.TButton'
        )
        self.record_button.bind('<ButtonPress-1>', self.start_recording)
        self.record_button.bind('<ButtonRelease-1>', self.stop_recording)
        self.record_button.pack(pady=10)

        # Status label
        self.status_label = ttk.Label(
            left_panel,
            text="Ready",
            font=("Arial", 12)
        )
        self.status_label.pack(pady=5)

        # Command list
        ttk.Label(left_panel, text="Available Commands:", font=("Arial", 12)).pack(pady=5)
        self.commands_text = tk.Text(left_panel, height=10, width=30, wrap=tk.WORD)
        self.commands_text.pack(pady=5, fill=tk.X)
        self.update_commands_list()

        # Middle panel for recordings
        middle_panel = ttk.Frame(main_frame)
        middle_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Training section
        train_frame = ttk.LabelFrame(middle_panel, text="Command Training")
        train_frame.pack(fill=tk.X, pady=5)

        # Command entry
        ttk.Label(train_frame, text="Command text:").pack(pady=2)
        self.command_entry = ttk.Entry(train_frame)
        self.command_entry.pack(fill=tk.X, padx=5, pady=2)

        # Train button
        train_button = ttk.Button(
            train_frame, 
            text="Save Recording with Command",
            command=self.save_command
        )
        train_button.pack(pady=5)

        # Add Bulk Record Button
        bulk_record_button = ttk.Button(
            train_frame,
            text="Bulk Record Command (20-30x)",
            command=self.bulk_record_command
        )
        bulk_record_button.pack(pady=5, padx=5, fill=tk.X)

        # Recordings list
        ttk.Label(middle_panel, text="Recent Recordings:", font=("Arial", 12)).pack(pady=5)
        
        # Recordings list with scrollbar
        self.recordings_list = tk.Listbox(middle_panel, height=15)
        scrollbar = ttk.Scrollbar(middle_panel, orient="vertical", command=self.recordings_list.yview)
        self.recordings_list.configure(yscrollcommand=scrollbar.set)
        
        self.recordings_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Buttons frame
        buttons_frame = ttk.Frame(middle_panel)
        buttons_frame.pack(fill=tk.X, pady=5)

        # Test button
        test_button = ttk.Button(
            buttons_frame, 
            text="Test Selected Recording",
            command=self.test_selected_recording
        )
        test_button.pack(side=tk.LEFT, padx=5)

        # Delete button
        delete_button = ttk.Button(
            buttons_frame,
            text="Delete Selected",
            command=self.delete_selected_recording
        )
        delete_button.pack(side=tk.RIGHT, padx=5)

        # Right panel for background operations
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # Background operations
        bg_frame = ttk.LabelFrame(right_panel, text="Background Operations")
        bg_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Background operations text
        self.bg_text = tk.Text(
            bg_frame,
            height=15,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        bg_scrollbar = ttk.Scrollbar(bg_frame, orient="vertical", command=self.bg_text.yview)
        self.bg_text.configure(yscrollcommand=bg_scrollbar.set)
        
        self.bg_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        bg_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def add_background_operation(self, text):
        """Add text to background operations log"""
        self.bg_text.config(state=tk.NORMAL)
        self.bg_text.insert(tk.END, f"{text}\n")
        self.bg_text.see(tk.END)
        self.bg_text.config(state=tk.DISABLED)
        self.root.update()

    def update_commands_list(self):
        """Update the list of available commands"""
        commands = self.command_organizer.get_all_commands()
        self.commands_text.delete('1.0', tk.END)
        for cmd in commands:
            self.commands_text.insert(tk.END, f"- {cmd}\n")

    def start_recording(self, event=None):
        try:
            self.record_button.config(style='Recording.TButton')
            self.status_label.config(text="Recording...")
            self.root.update()
            
            self.add_background_operation("Starting audio recording...")
            if not self.audio_recorder.start_recording_async():
                raise Exception("Failed to start recording")
                
        except Exception as e:
            messagebox.showerror("Recording Error", f"Error starting recording: {str(e)}")
            self.status_label.config(text="Error")
            self.record_button.config(style='Record.TButton')

    def stop_recording(self, event=None):
        try:
            self.record_button.config(style='Record.TButton')
            self.status_label.config(text="Processing...")
            self.root.update()
            
            self.add_background_operation("Stopping recording...")
            # Stop recording and get the filename
            filename = self.audio_recorder.stop_recording_and_save()
            if filename:
                self.add_background_operation(f"Saved recording: {filename}")
                self.update_recordings_list()
                # Pre-fill command entry if we can recognize it
                threading.Thread(target=self.recognize_command, args=(filename,), daemon=True).start()
            else:
                raise Exception("Failed to save recording")
                
        except Exception as e:
            messagebox.showerror("Recording Error", f"Error stopping recording: {str(e)}")
            self.status_label.config(text="Error")

    def recognize_command(self, audio_file):
        """Try to recognize command from audio and pre-fill the command entry"""
        try:
            self.add_background_operation("Processing audio for command recognition...")

            # Ensure model and processor are loaded
            if not self.model_trainer.model or not self.model_trainer.processor:
                raise Exception("Model or processor is not loaded.")

            # Load and preprocess the audio
            waveform, _ = librosa.load(audio_file, sr=16000)
            
            # Process audio with feature extractor first
            input_features = self.model_trainer.processor.feature_extractor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt"
            )
            
            # Move to device
            input_values = input_features.input_values.to(self.model_trainer.device)

            # Get the model output - use same precision as model
            with torch.no_grad():
                if self.model_trainer.device.type == "cuda":
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        logits = self.model_trainer.model(input_values).logits
                else:
                    logits = self.model_trainer.model(input_values).logits

            # Decode the output using tokenizer directly
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.model_trainer.processor.tokenizer.decode(predicted_ids[0].tolist())
            
            # Clean up the transcription
            transcription = transcription.lower().strip()
            
            self.command_entry.delete(0, tk.END)
            self.command_entry.insert(0, transcription)
            self.add_background_operation(f"Recognized command: {transcription}")

        except Exception as e:
            self.add_background_operation(f"Error recognizing command: {str(e)}")
            print(f"Recognition error details: {str(e)}")  # More detailed error logging

    def save_command(self):
        """Save the recording with its command"""
        selection = self.recordings_list.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a recording to save")
            return

        command_text = self.command_entry.get().strip().lower()
        if not command_text:
            messagebox.showinfo("Info", "Please enter the command text")
            return

        recording_name = self.recordings_list.get(selection[0])
        recording_path = str(Path("recordings") / recording_name)

        try:
            self.add_background_operation(f"Saving recording for command: {command_text}")
            new_path = self.command_organizer.add_recording(recording_path, command_text)
            self.add_background_operation(f"Saved recording to: {new_path}")
            
            # Update UI
            self.update_commands_list()
            self.command_entry.delete(0, tk.END)
            messagebox.showinfo("Success", "Command saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error saving command: {str(e)}")

    def bulk_record_command(self):
        """Record a specific command multiple times with manual start/stop"""
        command_dialog = tk.Toplevel(self.root)
        command_dialog.title("Bulk Recording Setup")
        command_dialog.geometry("400x500")  # Made taller for better button visibility
        command_dialog.transient(self.root)
        command_dialog.grab_set()

        # Main container frame
        main_frame = ttk.Frame(command_dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Command selection frame
        select_frame = ttk.LabelFrame(main_frame, text="Command Selection")
        select_frame.pack(fill=tk.X, pady=5)

        ttk.Label(select_frame, text="Enter command to train (e.g., 'open notepad'):").pack(pady=5)
        command_entry = ttk.Entry(select_frame)
        command_entry.pack(fill=tk.X, padx=5, pady=5)
        
        # Number of recordings frame
        num_frame = ttk.LabelFrame(main_frame, text="Number of Recordings")
        num_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(num_frame, text="Number of recordings (20-30):").pack(pady=5)
        num_recordings = ttk.Scale(num_frame, from_=20, to=30, orient=tk.HORIZONTAL)
        num_recordings.set(25)
        num_recordings.pack(fill=tk.X, padx=5, pady=5)

        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="Recording Progress")
        progress_frame.pack(fill=tk.X, pady=5)
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(
            progress_frame,
            variable=progress_var,
            maximum=100
        )
        progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        status_label = ttk.Label(
            progress_frame, 
            text="Ready to start",
            font=("Arial", 10)
        )
        status_label.pack(pady=5)

        # Recording controls frame
        controls_frame = ttk.LabelFrame(main_frame, text="Recording Controls")
        controls_frame.pack(fill=tk.BOTH, expand=True, pady=5)  # Made expandable
        
        record_status_label = ttk.Label(
            controls_frame, 
            text="Click 'Begin Session' then use\nRecord/Stop for each sample", 
            font=("Arial", 11, "bold"),
            justify=tk.CENTER
        )
        record_status_label.pack(pady=10)
        
        # Large record button with clear styling
        record_button = ttk.Button(
            controls_frame, 
            text="Start Recording",
            style='BigRecord.TButton',
            state='disabled'  # Initially disabled
        )
        record_button.pack(pady=15, padx=20, fill=tk.X)

        def start_bulk_recording():
            command_text = command_entry.get().strip().lower()
            if not command_text:
                messagebox.showinfo("Info", "Please enter the command text")
                return

            num = int(num_recordings.get())
            current_recording = [0]  # Use list to allow modification in nested function
            is_recording = [False]   # Track recording state
            
            # Enable recording button and disable start button
            record_button.config(state='normal')
            start_btn.config(state='disabled')
            command_entry.config(state='disabled')
            num_recordings.config(state='disabled')
            
            def toggle_recording():
                if not is_recording[0]:
                    # Start recording
                    is_recording[0] = True
                    record_button.config(
                        text="◼ Stop Recording",
                        style='Recording.TButton'
                    )
                    record_status_label.config(
                        text=f"Recording {current_recording[0] + 1}/{num}...\nSpeak now!"
                    )
                    self.audio_recorder.start_recording_async()
                else:
                    # Stop recording
                    is_recording[0] = False
                    filename = self.audio_recorder.stop_recording_and_save()
                    
                    if filename:
                        # Save recording with command
                        self.command_organizer.add_recording(filename, command_text)
                        self.add_background_operation(f"Saved recording {current_recording[0] + 1}/{num}")
                        current_recording[0] += 1
                        progress = (current_recording[0] / num) * 100
                        progress_var.set(progress)
                        
                        if current_recording[0] >= num:
                            # All recordings complete
                            record_button.config(state='disabled')
                            record_status_label.config(
                                text="Recording complete!\nYou can close this window."
                            )
                            self.update_commands_list()
                            messagebox.showinfo("Success", f"Bulk recording completed! Recorded {num} samples.")
                            command_dialog.protocol("WM_DELETE_WINDOW", command_dialog.destroy)
                        else:
                            record_button.config(
                                text="⏺ Start Next Recording",
                                style='Action.TButton'
                            )
                            record_status_label.config(
                                text=f"Ready for recording {current_recording[0] + 1}/{num}\nClick Start when ready"
                            )

            record_button.config(command=toggle_recording)

        # Start button at the bottom
        start_btn = ttk.Button(
            main_frame,
            text="Begin Recording Session",
            command=start_bulk_recording,
            style='Action.TButton'
        )
        start_btn.pack(pady=10, padx=20, fill=tk.X)

        # Create special styles for the buttons
        style = ttk.Style()
        style.configure('Action.TButton', 
            padding=10, 
            font=('Arial', 10, 'bold')
        )
        style.configure('BigRecord.TButton', 
            padding=15,
            font=('Arial', 12, 'bold')
        )
        style.configure('Recording.TButton', 
            padding=15,
            font=('Arial', 12, 'bold')
        )

        # Center the dialog
        command_dialog.update_idletasks()
        command_dialog.geometry(f"+{self.root.winfo_x() + 100}+{self.root.winfo_y() + 100}")
        
        # Set initial focus to the command entry
        command_entry.focus()

    def update_recordings_list(self):
        """Update the list of recordings"""
        self.recordings_list.delete(0, tk.END)
        recordings = self.audio_recorder.get_recordings_list()
        for recording in recordings:
            self.recordings_list.insert(tk.END, os.path.basename(recording))

    def test_selected_recording(self):
        """Test the selected recording"""
        selection = self.recordings_list.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a recording to test")
            return
            
        recording_name = self.recordings_list.get(selection[0])
        recording_path = str(Path("recordings") / recording_name)
        
        if os.path.exists(recording_path):
            self.status_label.config(text="Processing recording...")
            self.root.update()
            threading.Thread(target=self.recognize_command, args=(recording_path,), daemon=True).start()
        else:
            messagebox.showerror("Error", "Recording file not found")

    def delete_selected_recording(self):
        """Delete the selected recording"""
        selection = self.recordings_list.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a recording to delete")
            return

        recording_name = self.recordings_list.get(selection[0])
        recording_path = str(Path("recordings") / recording_name)

        try:
            if os.path.exists(recording_path):
                os.remove(recording_path)
                self.update_recordings_list()
                self.add_background_operation(f"Deleted recording: {recording_path}")
                messagebox.showinfo("Success", "Recording deleted successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Error deleting recording: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    
    # Create custom styles
    style = ttk.Style()
    style.configure('Record.TButton', padding=10)
    style.configure('Recording.TButton', padding=10, background='red')
    
    # Note: This direct initialization is for testing only
    # Normally the app should be started through startup.py
    from model_trainer import ModelTrainer
    from command_organizer import CommandOrganizer
    
    model_trainer = ModelTrainer()
    command_organizer = CommandOrganizer()
    app = VoiceCommandApp(root, model_trainer, command_organizer)
    root.mainloop()