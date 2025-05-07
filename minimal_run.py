import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
import librosa
import time
import os
from model_trainer import ModelTrainer
from command_organizer import CommandOrganizer
from command_executor import CommandExecutor
from audio_recorder import AudioRecorder

class MinimalRunInterface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Voice Command Minimal Runner")
        self.root.geometry("550x600")  # Made wider and taller for queue display

        # Initialize with performance optimization settings
        self.use_online_services = False  # Set to False to disable online services
        self.model_trainer = ModelTrainer()
        self.model_trainer.optimize_memory = True  # Ensure memory optimization is enabled
        self.model_trainer.use_online_service = self.use_online_services
        
        self.command_organizer = CommandOrganizer()
        self.audio_recorder = AudioRecorder()
        self.command_executor = CommandExecutor()
        
        # Set command queue to minimal delay
        self.command_executor.command_queue.default_delay = 0.1  # Very minimal delay
        
        # Connect audio recorder to command executor
        self.command_executor.set_audio_recorder(self.audio_recorder)
        
        # Setup command queue callbacks
        self.command_executor.register_queue_callback("on_queue_update", self.update_queue_display)
        self.command_executor.register_queue_callback("on_command_start", self.on_command_start)
        self.command_executor.register_queue_callback("on_command_complete", self.on_command_complete)
        
        self.listening = False
        self.listen_thread = None
        
        # Status tracking
        self.status_messages = []
        self.max_status_messages = 10
        
        # Performance tracking
        self.last_transcription_time = 0
        self.transcription_times = []

        # Bulk training mode
        self.bulk_training_active = False
        self.bulk_training_command = None

        self.setup_gui()
        self.load_commands()

    def setup_gui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create notebook (tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Command Execution
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="Command Execution")
        
        # Tab 2: Training
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="Training")
        
        # Setup Tab 1 (Command Execution)
        self.setup_execution_tab(self.tab1)
        
        # Setup Tab 2 (Training)
        self.setup_training_tab(self.tab2)

    def setup_execution_tab(self, parent):
        # Status label
        self.status_label = ttk.Label(parent, text="Ready", font=("Arial", 12))
        self.status_label.pack(pady=5)

        # Listen controls
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(pady=5)
        self.start_btn = ttk.Button(btn_frame, text="Start Listening", command=self.start_listening)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="Stop Listening", command=self.stop_listening, state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.clear_queue_btn = ttk.Button(btn_frame, text="Clear Queue", command=self.clear_command_queue)
        self.clear_queue_btn.pack(side=tk.LEFT, padx=5)

        # Online services toggle
        self.online_var = tk.BooleanVar(value=self.use_online_services)
        self.online_check = ttk.Checkbutton(
            btn_frame, 
            text="Online Services", 
            variable=self.online_var,
            command=self.toggle_online_services
        )
        self.online_check.pack(side=tk.LEFT, padx=10)

        # Command Queue Frame
        queue_frame = ttk.LabelFrame(parent, text="Command Queue")
        queue_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # Current command
        current_frame = ttk.Frame(queue_frame)
        current_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(current_frame, text="Executing:").pack(side=tk.LEFT, padx=5)
        self.current_cmd_label = ttk.Label(current_frame, text="None", style="Current.TLabel")
        self.current_cmd_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Queue list
        ttk.Label(queue_frame, text="Upcoming Commands:").pack(anchor=tk.W, padx=5)
        self.queue_listbox = tk.Listbox(queue_frame, height=5)
        self.queue_listbox.pack(fill=tk.X, padx=5, pady=5)

        # Manual command buttons - in a scrollable frame
        cmd_outer_frame = ttk.LabelFrame(parent, text="Manual Commands")
        cmd_outer_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Add canvas with scrollbar for commands
        self.cmd_canvas = tk.Canvas(cmd_outer_frame)
        scrollbar = ttk.Scrollbar(cmd_outer_frame, orient="vertical", command=self.cmd_canvas.yview)
        self.cmd_canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.cmd_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Frame inside canvas for buttons
        self.cmds_frame = ttk.Frame(self.cmd_canvas)
        self.cmd_canvas.create_window((0, 0), window=self.cmds_frame, anchor="nw", tags="self.cmds_frame")
        
        # Configure scrolling
        self.cmds_frame.bind("<Configure>", self.on_frame_configure)
        
        # Status Log
        log_frame = ttk.LabelFrame(parent, text="Status Log")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.log_text = tk.Text(log_frame, height=6, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_training_tab(self, parent):
        # Command selection frame
        cmd_select_frame = ttk.Frame(parent)
        cmd_select_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(cmd_select_frame, text="Command:").pack(side=tk.LEFT, padx=5)
        self.command_var = tk.StringVar()
        self.command_combo = ttk.Combobox(cmd_select_frame, textvariable=self.command_var)
        self.command_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Button to add new command
        ttk.Button(cmd_select_frame, text="New Command", command=self.add_new_command).pack(side=tk.LEFT, padx=5)
        
        # Bulk training controls
        bulk_frame = ttk.LabelFrame(parent, text="Bulk Training")
        bulk_frame.pack(fill=tk.X, pady=10, padx=5)
        
        bulk_btn_frame = ttk.Frame(bulk_frame)
        bulk_btn_frame.pack(fill=tk.X, pady=5)
        
        self.start_bulk_btn = ttk.Button(bulk_btn_frame, text="Start Bulk Training", 
                                         command=self.start_bulk_training)
        self.start_bulk_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_bulk_btn = ttk.Button(bulk_btn_frame, text="Stop Bulk Training",
                                        command=self.stop_bulk_training, state='disabled')
        self.stop_bulk_btn.pack(side=tk.LEFT, padx=5)
        
        self.record_sample_btn = ttk.Button(bulk_btn_frame, text="Record Sample",
                                           command=self.record_training_sample)
        self.record_sample_btn.pack(side=tk.LEFT, padx=5)
        
        # Bulk training status
        bulk_status_frame = ttk.Frame(bulk_frame)
        bulk_status_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(bulk_status_frame, text="Status:").pack(side=tk.LEFT, padx=5)
        self.bulk_status_label = ttk.Label(bulk_status_frame, text="Idle")
        self.bulk_status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Label(bulk_status_frame, text="Recordings:").pack(side=tk.LEFT, padx=5)
        self.bulk_count_label = ttk.Label(bulk_status_frame, text="0")
        self.bulk_count_label.pack(side=tk.LEFT, padx=5)
        
        # Training management
        manage_frame = ttk.LabelFrame(parent, text="Training Management")
        manage_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)
        
        ttk.Button(manage_frame, text="Train Model", 
                   command=self.train_model).pack(anchor=tk.W, padx=5, pady=5)
                   
        ttk.Button(manage_frame, text="Import Recordings", 
                   command=self.import_recordings).pack(anchor=tk.W, padx=5, pady=5)
                   
        # Training status
        self.training_progress = ttk.Progressbar(manage_frame, orient="horizontal", mode="determinate")
        self.training_progress.pack(fill=tk.X, padx=5, pady=5)
        
        self.training_status_label = ttk.Label(manage_frame, text="Not training")
        self.training_status_label.pack(anchor=tk.W, padx=5)
        
        # Command stats
        stats_frame = ttk.Frame(manage_frame)
        stats_frame.pack(fill=tk.X, pady=10)
        
        self.stats_tree = ttk.Treeview(stats_frame, columns=("command", "samples"), show="headings")
        self.stats_tree.heading("command", text="Command")
        self.stats_tree.heading("samples", text="Samples")
        self.stats_tree.column("command", width=150)
        self.stats_tree.column("samples", width=50)
        
        scrollbar = ttk.Scrollbar(stats_frame, orient="vertical", command=self.stats_tree.yview)
        self.stats_tree.configure(yscrollcommand=scrollbar.set)
        
        self.stats_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def on_frame_configure(self, event=None):
        """Reset the scroll region to encompass the inner frame"""
        self.cmd_canvas.configure(scrollregion=self.cmd_canvas.bbox("all"))
    
    def setup_styles(self):
        """Setup custom styles for the interface"""
        style = ttk.Style()
        style.configure("Current.TLabel", foreground="blue", font=("Arial", 10, "bold"))
        style.configure("TButton", padding=5)

    def toggle_online_services(self):
        """Toggle between online and local transcription services"""
        self.use_online_services = self.online_var.get()
        self.model_trainer.use_online_service = self.use_online_services
        
        if self.use_online_services:
            self.add_log_message("Switched to online transcription services")
            # Free memory by cleaning up local model
            if self.model_trainer.model:
                self.model_trainer.cleanup()
        else:
            self.add_log_message("Switched to local transcription model")
            
    def load_commands(self):
        """Load commands from model_trainer and update UI"""
        # Load model commands
        self.model_trainer.load_trained_model()
        commands = sorted(self.model_trainer.commands)
        
        # Update command execution buttons
        for widget in self.cmds_frame.winfo_children():
            widget.destroy()
            
        # Create buttons for all commands
        for i, cmd in enumerate(commands):
            btn = ttk.Button(self.cmds_frame, text=cmd.title(), command=lambda c=cmd: self.run_command(c))
            btn.grid(row=i//2, column=i%2, sticky="ew", padx=2, pady=2)
        
        # Update canvas scroll region
        self.cmds_frame.update_idletasks()
        self.cmd_canvas.configure(scrollregion=self.cmd_canvas.bbox("all"))
        
        # Update command dropdown in training tab
        self.command_combo['values'] = commands
        if commands:
            self.command_combo.current(0)
            
        # Update command statistics
        self.update_command_stats()
        
        # Add initial log message
        self.add_log_message(f"System initialized with {'online' if self.use_online_services else 'local'} transcription")

    def update_command_stats(self):
        """Update the command statistics in the training tab"""
        # Clear existing stats
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
            
        # Add current command stats
        commands = self.command_organizer.get_all_commands()
        commands.sort()
        
        for cmd in commands:
            recordings = self.command_organizer.get_recordings_for_command(cmd)
            self.stats_tree.insert("", "end", values=(cmd, len(recordings)))
        
        # Add commands that exist in model but have no recordings
        for cmd in self.model_trainer.commands:
            if cmd not in commands:
                self.stats_tree.insert("", "end", values=(cmd, 0))

    def add_new_command(self):
        """Add a new command to the system"""
        command_text = simpledialog.askstring("New Command", "Enter the new command text:")
        if command_text:
            # Clean and normalize the command text
            command_text = command_text.lower().strip()
            
            # Create command folder
            folder_path = self.command_organizer.create_new_command(command_text)
            
            # Update UI
            self.add_log_message(f"Created new command: {command_text}")
            self.load_commands()
            
            # Set as current command
            self.command_var.set(command_text)

    def start_bulk_training(self):
        """Start bulk training mode for the selected command"""
        command_text = self.command_var.get()
        if not command_text:
            messagebox.showerror("Error", "Please select or create a command first")
            return
        
        # Start bulk training mode
        result = self.command_organizer.start_bulk_training(command_text)
        if result:
            self.bulk_training_active = True
            self.bulk_training_command = command_text
            
            # Update UI
            self.start_bulk_btn.config(state='disabled')
            self.stop_bulk_btn.config(state='normal')
            self.bulk_status_label.config(text=f"Training: {command_text}")
            self.bulk_count_label.config(text="0")
            
            self.add_log_message(f"Started bulk training for '{command_text}'")

    def stop_bulk_training(self):
        """Stop bulk training mode"""
        if self.bulk_training_active:
            result = self.command_organizer.stop_bulk_training()
            
            # Update UI
            self.start_bulk_btn.config(state='normal')
            self.stop_bulk_btn.config(state='disabled')
            self.bulk_status_label.config(text="Idle")
            
            self.add_log_message(f"Stopped bulk training for '{result['command']}'. "
                                f"Processed {result['recordings_processed']} recordings.")
            
            self.bulk_training_active = False
            self.bulk_training_command = None
            
            # Update command stats
            self.update_command_stats()

    def record_training_sample(self):
        """Record a training sample for bulk training or regular training"""
        command_text = self.command_var.get()
        if not command_text:
            messagebox.showerror("Error", "Please select or create a command first")
            return
        
        # Start recording
        self.bulk_status_label.config(text=f"Recording for: {command_text}")
        if not self.audio_recorder.start_recording_async():
            self.add_log_message("Error starting recording")
            return
        
        # Wait for recording to complete (1.5 seconds)
        for i in range(15):
            self.root.update()
            time.sleep(0.1)
        
        # Stop recording and save
        filename = self.audio_recorder.stop_recording_and_save()
        
        if filename:
            # Process the recording
            if self.bulk_training_active:
                # Add to bulk training
                self.command_organizer.process_recording_for_bulk_training(filename)
                count = self.command_organizer.bulk_recordings_processed
                self.bulk_count_label.config(text=str(count))
                self.add_log_message(f"Added recording {count} to '{self.bulk_training_command}'")
            else:
                # Add as regular recording
                self.command_organizer.add_recording(filename, command_text)
                self.add_log_message(f"Added recording to command '{command_text}'")
                self.update_command_stats()
        else:
            self.add_log_message("Failed to record audio sample")
            
        # Reset status
        if self.bulk_training_active:
            self.bulk_status_label.config(text=f"Training: {self.bulk_training_command}")
        else:
            self.bulk_status_label.config(text="Idle")

    def train_model(self):
        """Train the model using the current command data"""
        # Start training in background
        self.training_status_label.config(text="Training...")
        self.training_progress["value"] = 0
        
        threading.Thread(target=self._do_train_model, daemon=True).start()

    def _do_train_model(self):
        """Background thread to perform model training"""
        try:
            # Get command folders
            command_folders = self.command_organizer.get_command_folders()
            
            if not command_folders:
                self.add_log_message("No command data available for training")
                self.training_status_label.config(text="No data for training")
                return
                
            # Update progress
            self.add_log_message(f"Starting model training with {len(command_folders)} commands")
            self.training_progress["value"] = 10
            self.root.update_idletasks()
            
            # Train model
            success = self.model_trainer.train_model(command_folders)
            
            # Update progress
            self.training_progress["value"] = 100
            
            if success:
                self.add_log_message("Model training complete")
                self.training_status_label.config(text="Training complete")
                # Reload commands to update UI
                self.load_commands()
            else:
                self.add_log_message("Model training failed")
                self.training_status_label.config(text="Training failed")
                
        except Exception as e:
            self.add_log_message(f"Error in model training: {e}")
            self.training_status_label.config(text=f"Error: {str(e)}")

    def import_recordings(self):
        """Import recordings from another folder"""
        # To be implemented - would use file dialog to select folder
        self.add_log_message("Import recordings feature not implemented yet")

    def run_command(self, cmd):
        success = self.command_executor.execute_command(cmd)
        self.add_log_message(f"Command: {cmd} - {'Queued' if success else 'Failed'}")
        if success:
            self.status_label.config(text=f"Queued: {cmd}")
        else:
            self.status_label.config(text=f"Failed: {cmd}")

    def update_queue_display(self, queue_status):
        """Update the queue display with current queue status"""
        try:
            # Update current command label
            current_cmd = queue_status.get("current_command", "None")
            self.current_cmd_label.config(text=current_cmd if current_cmd else "None")
            
            # Update queue list
            self.queue_listbox.delete(0, tk.END)
            for cmd in queue_status.get("next_commands", []):
                self.queue_listbox.insert(tk.END, cmd)
                
            # Update root to ensure display refreshes
            self.root.update_idletasks()
        except Exception as e:
            print(f"Error updating queue display: {e}")
            
    def on_command_start(self, command):
        """Called when a command starts executing"""
        self.add_log_message(f"Executing: {command}")
        self.status_label.config(text=f"Executing: {command}")
        
    def on_command_complete(self, command):
        """Called when a command completes execution"""
        self.add_log_message(f"Completed: {command}")
        
    def clear_command_queue(self):
        """Clear the command queue"""
        self.command_executor.execute_command("clear queue")
        self.add_log_message("Command queue cleared")
        
    def start_listening(self):
        if self.listening:
            return
        self.listening = True
        self.status_label.config(text="Listening for voice command...")
        self.add_log_message("Started listening for commands")
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.listen_thread = threading.Thread(target=self.listen_loop, daemon=True)
        self.listen_thread.start()

    def stop_listening(self):
        self.listening = False
        self.status_label.config(text="Stopped listening.")
        self.add_log_message("Stopped listening")
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
        # Log average transcription time
        if self.transcription_times:
            avg_time = sum(self.transcription_times) / len(self.transcription_times)
            self.add_log_message(f"Average transcription time: {avg_time:.2f}s")
            self.transcription_times = []

    def listen_loop(self):
        while self.listening:
            # Record one command
            self.status_label.config(text="Listening... Speak now!")
            self.root.update()
            if not self.audio_recorder.start_recording_async():
                self.status_label.config(text="Error starting recording")
                self.add_log_message("Error starting audio recording")
                break
                
            # Wait for 2 seconds (or until stop)
            for _ in range(20):
                if not self.listening:
                    self.audio_recorder.stop_recording_and_save()
                    return
                self.root.after(100)
                self.root.update()
                
            filename = self.audio_recorder.stop_recording_and_save()
            if filename:
                # Start timing transcription
                start_time = time.time()
                
                waveform, _ = librosa.load(filename, sr=16000)
                cmd = self.model_trainer.transcribe_audio(waveform, audio_file=filename)
                
                # End timing and record
                transcription_time = time.time() - start_time
                self.transcription_times.append(transcription_time)
                self.add_log_message(f"Transcription took {transcription_time:.2f}s")
                
                if cmd:
                    if isinstance(cmd, list):
                        # Multiple commands detected
                        self.status_label.config(text=f"Recognized {len(cmd)} commands")
                        self.add_log_message(f"Multiple commands: {', '.join(cmd)}")
                    else:
                        self.status_label.config(text=f"Recognized: {cmd}")
                        self.add_log_message(f"Recognized: {cmd}")
                        
                    self.run_command(cmd)
                else:
                    self.status_label.config(text="Could not recognize command")
                    self.add_log_message("Could not recognize command")
            else:
                self.status_label.config(text="No audio recorded")
                self.add_log_message("No audio recorded")
                
            # Wait a bit before next listen - reduced delay
            for _ in range(5):  # Short delay (0.5s) between listening attempts
                if not self.listening:
                    return
                self.root.after(100)
                self.root.update()
                
    def add_log_message(self, message):
        """Add a message to the log display"""
        try:
            timestamp = time.strftime("%H:%M:%S")
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
            
            # Also print to console for debugging
            print(f"[{timestamp}] {message}")
            
            # Keep track of status messages
            self.status_messages.append(message)
            if len(self.status_messages) > self.max_status_messages:
                self.status_messages.pop(0)
        except Exception as e:
            print(f"Error adding log message: {e}")

    def run(self):
        self.setup_styles()
        self.root.mainloop()

if __name__ == "__main__":
    app = MinimalRunInterface()
    app.run()
