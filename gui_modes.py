import tkinter as tk
from tkinter import ttk, messagebox
import threading
from pathlib import Path
import torch
import librosa
from audio_recorder import AudioRecorder
from command_organizer import CommandOrganizer
from command_executor import CommandExecutor

class RecordingMode:
    def __init__(self, root, command_organizer):
        self.root = root
        self.root.title("Voice Command System - Recording Mode")
        self.command_organizer = command_organizer
        self.audio_recorder = AudioRecorder()
        self.setup_gui()
        
    def setup_gui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Single recording section
        record_frame = ttk.LabelFrame(main_frame, text="Quick Recording")
        record_frame.pack(fill=tk.X, pady=5)
        
        self.record_button = ttk.Button(
            record_frame,
            text="Press and Hold to Record",
            style='Record.TButton'
        )
        self.record_button.bind('<ButtonPress-1>', self.start_recording)
        self.record_button.bind('<ButtonRelease-1>', self.stop_recording)
        self.record_button.pack(pady=10)
        
        self.status_label = ttk.Label(
            record_frame,
            text="Ready",
            font=("Arial", 12)
        )
        self.status_label.pack(pady=5)
        
        # Bulk recording section
        bulk_frame = ttk.LabelFrame(main_frame, text="Bulk Recording")
        bulk_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            bulk_frame,
            text="Start Bulk Recording Session",
            command=self.bulk_record_command
        ).pack(pady=10, padx=20, fill=tk.X)
        
        # Recordings list
        list_frame = ttk.LabelFrame(main_frame, text="Recent Recordings")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.recordings_list = tk.Listbox(list_frame, height=10)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.recordings_list.yview)
        self.recordings_list.configure(yscrollcommand=scrollbar.set)
        
        self.recordings_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Control buttons
        btn_frame = ttk.Frame(list_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Delete Selected", command=self.delete_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Play Selected", command=self.play_selected).pack(side=tk.RIGHT, padx=5)
        
        self.update_recordings_list()
        
    def start_recording(self, event=None):
        try:
            self.record_button.config(style='Recording.TButton')
            self.status_label.config(text="Recording...")
            self.root.update()
            
            if not self.audio_recorder.start_recording_async():
                raise Exception("Failed to start recording")
                
        except Exception as e:
            messagebox.showerror("Recording Error", str(e))
            self.status_label.config(text="Error")
            self.record_button.config(style='Record.TButton')
    
    def stop_recording(self, event=None):
        try:
            self.record_button.config(style='Record.TButton')
            self.status_label.config(text="Processing...")
            self.root.update()
            
            filename = self.audio_recorder.stop_recording_and_save()
            if filename:
                self.status_label.config(text="Recording saved")
                self.update_recordings_list()
            else:
                raise Exception("Failed to save recording")
                
        except Exception as e:
            messagebox.showerror("Recording Error", str(e))
            self.status_label.config(text="Error")
    
    def bulk_record_command(self):
        # Open bulk recording dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Bulk Recording")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        ttk.Label(main_frame, text="Command to record:").pack(pady=5)
        command_entry = ttk.Entry(main_frame)
        command_entry.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(main_frame, text="Number of recordings:").pack(pady=5)
        num_var = tk.IntVar(value=25)
        ttk.Scale(main_frame, from_=20, to=30, variable=num_var, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5)
        
        ttk.Button(
            main_frame,
            text="Start Recording Session",
            command=lambda: self.start_bulk_session(dialog, command_entry.get(), num_var.get())
        ).pack(pady=10)
    
    def start_bulk_session(self, dialog, command, num_recordings):
        if not command.strip():
            messagebox.showwarning("Input Required", "Please enter the command to record")
            return
            
        dialog.destroy()
        
        # Create bulk recording dialog
        session = tk.Toplevel(self.root)
        session.title(f"Recording Session: {command}")
        session.transient(self.root)
        session.grab_set()
        
        main_frame = ttk.Frame(session)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Progress information
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
            text="Click Record to start",
            font=("Arial", 11),
            justify=tk.CENTER
        )
        status_label.pack(pady=5)
        
        # Recording controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        current = [0]  # Current recording count
        is_recording = [False]  # Recording state
        
        def toggle_recording():
            if not is_recording[0]:
                # Start recording
                is_recording[0] = True
                record_btn.config(
                    text="◼ Stop Recording",
                    style='Recording.TButton'
                )
                status_label.config(
                    text=f"Recording {current[0] + 1}/{num_recordings}...\nSpeak now!"
                )
                self.audio_recorder.start_recording_async()
            else:
                # Stop recording
                is_recording[0] = False
                record_btn.config(
                    text="⏺ Start Next Recording",
                    style='Record.TButton'
                )
                
                filename = self.audio_recorder.stop_recording_and_save()
                if filename:
                    # Save recording with command
                    self.command_organizer.add_recording(filename, command)
                    current[0] += 1
                    
                    # Update progress
                    progress = (current[0] / num_recordings) * 100
                    progress_var.set(progress)
                    
                    if current[0] >= num_recordings:
                        # All recordings complete
                        record_btn.config(state='disabled')
                        status_label.config(
                            text="Recording session complete!\nYou can close this window."
                        )
                        self.update_recordings_list()
                        messagebox.showinfo(
                            "Success",
                            f"Bulk recording completed! Recorded {num_recordings} samples."
                        )
                    else:
                        status_label.config(
                            text=f"Ready for recording {current[0] + 1}/{num_recordings}\nClick Record when ready"
                        )
        
        record_btn = ttk.Button(
            control_frame,
            text="⏺ Start Recording",
            style='Record.TButton',
            command=toggle_recording
        )
        record_btn.pack(pady=5, padx=20, fill=tk.X)
        
        # Center the dialog
        session.update_idletasks()
        session.geometry(f"+{self.root.winfo_x() + 50}+{self.root.winfo_y() + 50}")
    
    def update_recordings_list(self):
        self.recordings_list.delete(0, tk.END)
        recordings = self.audio_recorder.get_recordings_list()
        for recording in recordings:
            self.recordings_list.insert(tk.END, Path(recording).name)
    
    def delete_selected(self):
        selection = self.recordings_list.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a recording to delete")
            return
            
        recording_name = self.recordings_list.get(selection[0])
        recording_path = str(Path("recordings") / recording_name)
        
        try:
            if Path(recording_path).exists():
                Path(recording_path).unlink()
                self.update_recordings_list()
                messagebox.showinfo("Success", "Recording deleted successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Error deleting recording: {str(e)}")
    
    def play_selected(self):
        # Implement audio playback functionality
        pass

class TrainingMode:
    def __init__(self, root, model_trainer, command_organizer):
        self.root = root
        self.root.title("Voice Command System - Training Mode")
        self.model_trainer = model_trainer
        self.command_organizer = command_organizer
        self.setup_gui()
        
    def setup_gui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Data overview section
        data_frame = ttk.LabelFrame(main_frame, text="Training Data Overview")
        data_frame.pack(fill=tk.X, pady=5)
        
        self.data_text = tk.Text(data_frame, height=10, wrap=tk.WORD)
        self.data_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.update_data_overview()
        
        # Training controls
        control_frame = ttk.LabelFrame(main_frame, text="Training Controls")
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            control_frame,
            text="Start Training",
            command=self.start_training
        ).pack(pady=10, padx=20, fill=tk.X)
        
        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="Training Progress")
        progress_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_label = ttk.Label(
            progress_frame,
            text="Ready to train",
            wraplength=400
        )
        self.status_label.pack(pady=5)
        
    def update_data_overview(self):
        self.data_text.delete('1.0', tk.END)
        command_folders = self.command_organizer.get_command_folders()
        
        if not command_folders:
            self.data_text.insert(tk.END, "No training data available.\nPlease record some commands first.")
            return
            
        for folder in command_folders:
            name = folder.name
            count = len(list(folder.glob("*.wav")))
            self.data_text.insert(tk.END, f"Command '{name}': {count} recordings\n")
    
    def start_training(self):
        command_folders = self.command_organizer.get_command_folders()
        if not command_folders:
            messagebox.showwarning("No Data", "No training data available. Please record some commands first.")
            return
            
        self.status_label.config(text="Initializing training...")
        threading.Thread(target=self.train_model, daemon=True).start()
        threading.Thread(target=self.update_progress, daemon=True).start()
    
    def train_model(self):
        try:
            if not self.model_trainer.load_base_model():
                self.status_label.config(text="Failed to load base model")
                return
                
            command_folders = self.command_organizer.get_command_folders()
            success = self.model_trainer.train_model(command_folders)
            
            if success:
                self.status_label.config(text="Training completed successfully!")
            else:
                self.status_label.config(text="Training failed. Please check the logs.")
        except Exception as e:
            self.status_label.config(text=f"Error during training: {str(e)}")
    
    def update_progress(self):
        while True:
            try:
                progress = self.model_trainer.get_progress()
                self.progress_var.set(progress["progress"])
                self.status_label.config(text=progress["operation"])
                self.root.update()
                if progress["progress"] >= 100:
                    break
            except:
                break
            self.root.after(100)

class DeploymentMode:
    def __init__(self, root, model_trainer, command_organizer):
        self.root = root
        self.root.title("Voice Command System - Testing/Deployment")
        self.model_trainer = model_trainer
        self.command_organizer = command_organizer
        self.audio_recorder = AudioRecorder()
        self.command_executor = CommandExecutor()
        
        # Ensure model and commands are loaded
        if not self.model_trainer.model:
            self.model_trainer.load_trained_model()
        
        self.setup_gui()
        
    def setup_gui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Available commands section
        commands_frame = ttk.LabelFrame(main_frame, text="Available Commands")
        commands_frame.pack(fill=tk.X, pady=5)
        
        self.commands_text = tk.Text(commands_frame, height=8, wrap=tk.WORD)
        self.commands_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.update_commands_list()
        
        # Testing section
        test_frame = ttk.LabelFrame(main_frame, text="Command Testing")
        test_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.record_button = ttk.Button(
            test_frame,
            text="Hold to Record Test Command",
            style='Record.TButton'
        )
        self.record_button.bind('<ButtonPress-1>', self.start_recording)
        self.record_button.bind('<ButtonRelease-1>', self.stop_recording)
        self.record_button.pack(pady=10)
        
        self.result_label = ttk.Label(
            test_frame,
            text="Ready for testing",
            font=("Arial", 12)
        )
        self.result_label.pack(pady=5)
        
        # Results log
        log_frame = ttk.LabelFrame(main_frame, text="Test Results")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def update_commands_list(self):
        self.commands_text.delete('1.0', tk.END)
        commands = set()
        for folder in self.command_organizer.get_command_folders():
            commands.add(folder.name.replace('_', ' '))
        
        for cmd in sorted(commands):
            self.commands_text.insert(tk.END, f"• {cmd}\n")
    
    def start_recording(self, event=None):
        try:
            self.record_button.config(style='Recording.TButton')
            self.result_label.config(text="Recording test command...")
            self.root.update()
            
            if not self.audio_recorder.start_recording_async():
                raise Exception("Failed to start recording")
                
        except Exception as e:
            messagebox.showerror("Recording Error", str(e))
            self.result_label.config(text="Error")
            self.record_button.config(style='Record.TButton')
    
    def stop_recording(self, event=None):
        try:
            self.record_button.config(style='Record.TButton')
            self.result_label.config(text="Processing...")
            self.root.update()
            
            filename = self.audio_recorder.stop_recording_and_save()
            if filename:
                # Process the recording and show results
                self.process_test_recording(filename)
            else:
                raise Exception("Failed to save recording")
                
        except Exception as e:
            messagebox.showerror("Recording Error", str(e))
            self.result_label.config(text="Error")
    
    def process_test_recording(self, filename):
        try:
            self.result_label.config(text="Processing...")
            self.root.update()
            
            # Load and process audio
            waveform, _ = librosa.load(filename, sr=16000)
            
            # Use Whisper transcription
            command = self.model_trainer.transcribe_audio(waveform)
            
            if command:
                # Execute the command
                if self.command_executor.execute_command(command):
                    result = f"Executed command: {command}"
                else:
                    result = f"Recognized but couldn't execute: {command}"
            else:
                result = "Could not recognize command"
                
            # Update UI
            self.result_label.config(text=result)
            self.log_text.insert('1.0', f"{result}\n")
            self.log_text.see('1.0')
            
        except Exception as e:
            error_msg = f"Error processing recording: {str(e)}"
            self.result_label.config(text="Error processing")
            self.log_text.insert('1.0', f"{error_msg}\n")
            self.log_text.see('1.0')