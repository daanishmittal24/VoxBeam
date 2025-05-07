import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import librosa
import os
import json
from pathlib import Path
import time
from datetime import datetime
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from model_trainer import ModelTrainer
from command_organizer import CommandOrganizer
from command_executor import CommandExecutor
from audio_recorder import AudioRecorder

class MinimalRunInterface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Voice Command Minimal Runner")
        self.root.geometry("600x450")

        self.model_trainer = ModelTrainer()
        self.command_organizer = CommandOrganizer()
        self.command_executor = CommandExecutor()
        self.audio_recorder = AudioRecorder()
        self.listening = False
        self.listen_thread = None
        
        # Accuracy testing variables
        self.test_results = defaultdict(list)
        self.confusion_matrix = None
        self.current_test_command = None
        self.testing_in_progress = False
        self.test_samples = {}

        self.setup_gui()
        self.load_commands()

    def setup_gui(self):
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Main tab
        main_frame = ttk.Frame(self.notebook)
        self.notebook.add(main_frame, text="Run Commands")
        
        # Test tab
        test_frame = ttk.Frame(self.notebook)
        self.notebook.add(test_frame, text="Accuracy Test")
        
        # Set up main frame
        self.setup_main_frame(main_frame)
        
        # Set up test frame
        self.setup_test_frame(test_frame)

    def setup_main_frame(self, frame):
        # Status label
        self.status_label = ttk.Label(frame, text="Ready", font=("Arial", 12))
        self.status_label.pack(pady=5)

        # Listen controls
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=5)
        self.start_btn = ttk.Button(btn_frame, text="Start Listening", command=self.start_listening)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="Stop Listening", command=self.stop_listening, state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Manual command buttons
        self.cmds_frame = ttk.LabelFrame(frame, text="Manual Commands")
        self.cmds_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
    def setup_test_frame(self, frame):
        # Test controls frame
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Start test button
        self.test_btn = ttk.Button(control_frame, text="Start Accuracy Test", command=self.start_accuracy_test)
        self.test_btn.pack(side=tk.LEFT, padx=5)
        
        # Stop test button
        self.stop_test_btn = ttk.Button(control_frame, text="Stop Testing", command=self.stop_accuracy_test, state='disabled')
        self.stop_test_btn.pack(side=tk.LEFT, padx=5)
        
        # Export results button
        self.export_btn = ttk.Button(control_frame, text="Export Results", command=self.export_results)
        self.export_btn.pack(side=tk.RIGHT, padx=5)
        
        # Create a notebook for test results
        results_notebook = ttk.Notebook(frame)
        results_notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Summary tab
        summary_frame = ttk.Frame(results_notebook)
        results_notebook.add(summary_frame, text="Summary")
        
        # Progress frame
        self.progress_frame = ttk.LabelFrame(summary_frame, text="Test Progress")
        self.progress_frame.pack(fill=tk.X, pady=5)
        
        # Status label
        self.test_status = ttk.Label(self.progress_frame, text="Ready to test", font=("Arial", 10))
        self.test_status.pack(pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)
        
        # Current command label
        self.cmd_label = ttk.Label(self.progress_frame, text="", font=("Arial", 12, "bold"))
        self.cmd_label.pack(pady=5)
        
        # Results frame
        self.results_frame = ttk.LabelFrame(summary_frame, text="Test Results")
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Results text
        self.results_text = tk.Text(self.results_frame, height=10, wrap=tk.WORD)
        results_scroll = ttk.Scrollbar(self.results_frame, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scroll.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create confusion matrix tab
        matrix_frame = ttk.Frame(results_notebook)
        results_notebook.add(matrix_frame, text="Confusion Matrix")
        
        # This will hold the matplotlib figure
        self.matrix_container = ttk.Frame(matrix_frame)
        self.matrix_container.pack(fill=tk.BOTH, expand=True)

    def load_commands(self):
        # Load commands from model_trainer or command_organizer
        self.model_trainer.load_trained_model()
        commands = sorted(self.model_trainer.commands)
        for widget in self.cmds_frame.winfo_children():
            widget.destroy()
        for cmd in commands:
            btn = ttk.Button(self.cmds_frame, text=cmd.title(), command=lambda c=cmd: self.run_command(c))
            btn.pack(fill=tk.X, pady=2, padx=5)

    def run_command(self, cmd):
        success = self.command_executor.execute_command(cmd)
        if success:
            self.status_label.config(text=f"Executed: {cmd}")
        else:
            self.status_label.config(text=f"Failed: {cmd}")

    def start_listening(self):
        if self.listening:
            return
        self.listening = True
        self.status_label.config(text="Listening for voice command...")
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.listen_thread = threading.Thread(target=self.listen_loop, daemon=True)
        self.listen_thread.start()

    def stop_listening(self):
        self.listening = False
        self.status_label.config(text="Stopped listening.")
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')

    def listen_loop(self):
        while self.listening:
            # Record one command
            self.status_label.config(text="Listening... Speak now!")
            self.root.update()
            if not self.audio_recorder.start_recording_async():
                self.status_label.config(text="Error starting recording")
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
                waveform, _ = librosa.load(filename, sr=16000)
                cmd = self.model_trainer.transcribe_audio(waveform)
                if cmd:
                    self.status_label.config(text=f"Recognized: {cmd}")
                    self.run_command(cmd)
                else:
                    self.status_label.config(text="Could not recognize command")
            else:
                self.status_label.config(text="No audio recorded")
            # Wait a bit before next listen
            for _ in range(10):
                if not self.listening:
                    return
                self.root.after(100)
                self.root.update()
    
    def start_accuracy_test(self):
        """Start the accuracy testing process"""
        if not self.model_trainer.commands:
            messagebox.showerror("Error", "No commands loaded for testing")
            return
            
        # Check if we're already testing
        if self.testing_in_progress:
            return
            
        # Reset previous results
        self.test_results = defaultdict(list)
        self.test_samples = {}
        self.testing_in_progress = True
        
        # Update UI
        self.test_btn.config(state='disabled')
        self.stop_test_btn.config(state='normal')
        self.test_status.config(text="Starting test...")
        self.cmd_label.config(text="")
        self.results_text.delete('1.0', tk.END)
        self.progress_var.set(0)
        self.root.update()
        
        # Start test thread
        threading.Thread(target=self.run_accuracy_test, daemon=True).start()
    
    def stop_accuracy_test(self):
        """Stop ongoing accuracy test"""
        self.testing_in_progress = False
        self.test_status.config(text="Test stopped.")
        self.test_btn.config(state='normal')
        self.stop_test_btn.config(state='disabled')
        
    def run_accuracy_test(self):
        """Run the accuracy test on each command"""
        commands = sorted(list(self.model_trainer.commands))
        total_commands = len(commands)
        completed = 0
        
        # Tell user how to complete the test
        self.results_text.insert(tk.END, "ACCURACY TESTING INSTRUCTIONS:\n\n")
        self.results_text.insert(tk.END, "1. For each command shown, speak the command clearly\n")
        self.results_text.insert(tk.END, "2. Wait for the system to process your speech\n") 
        self.results_text.insert(tk.END, "3. Results will be collected automatically\n\n")
        self.results_text.insert(tk.END, "Starting test...\n\n")
        
        # Test each command multiple times
        samples_per_command = 5
        total_tests = total_commands * samples_per_command
        
        try:
            # Loop through each command
            for cmd_idx, cmd in enumerate(commands):
                self.current_test_command = cmd
                self.cmd_label.config(text=f"Speak: \"{cmd.upper()}\"")
                
                # Test each command multiple times
                for sample in range(samples_per_command):
                    if not self.testing_in_progress:
                        return
                        
                    # Update progress
                    current_test = cmd_idx * samples_per_command + sample + 1
                    progress = (current_test / total_tests) * 100
                    self.progress_var.set(progress)
                    self.test_status.config(text=f"Testing: {cmd} (Sample {sample+1}/{samples_per_command})")
                    
                    # Record the command  
                    self.results_text.insert(tk.END, f"Testing: {cmd} (Sample {sample+1})\n")
                    self.results_text.see(tk.END)
                    self.root.update()
                    
                    # Record audio
                    if not self.audio_recorder.start_recording_async():
                        self.results_text.insert(tk.END, "Error starting recording\n")
                        continue
                    
                    # Wait for audio
                    countdown = 3
                    while countdown > 0:
                        self.test_status.config(text=f"Speak now: {cmd} (Recording in {countdown}...)")
                        self.root.update()
                        time.sleep(1)
                        countdown -= 1
                        
                    self.test_status.config(text=f"RECORDING NOW: {cmd}")
                    self.root.update()
                    
                    # Record for 2 seconds
                    time.sleep(2)
                    
                    # Stop recording
                    filename = self.audio_recorder.stop_recording_and_save()
                    if not filename:
                        self.results_text.insert(tk.END, "Error recording audio\n")
                        continue
                    
                    # Process the audio
                    waveform, _ = librosa.load(filename, sr=16000)
                    recognized_cmd = self.model_trainer.transcribe_audio(waveform)
                    
                    # Store result
                    if recognized_cmd:
                        self.test_results[cmd].append(recognized_cmd)
                        self.results_text.insert(tk.END, f"  → Recognized as: {recognized_cmd}\n")
                        if recognized_cmd == cmd:
                            self.results_text.insert(tk.END, "  ✓ CORRECT\n")
                        else:
                            self.results_text.insert(tk.END, "  ✗ INCORRECT\n")
                    else:
                        self.test_results[cmd].append(None)
                        self.results_text.insert(tk.END, f"  → Not recognized\n")
                        
                    # Save the test sample
                    if cmd not in self.test_samples:
                        self.test_samples[cmd] = []
                    self.test_samples[cmd].append(filename)
                    
                    completed += 1
                    self.results_text.see(tk.END)
                    self.root.update()
                    
                    # Pause between samples
                    time.sleep(1)
                
            # Test completed
            self.test_status.config(text="Test completed!")
            self.results_text.insert(tk.END, "\nTesting completed. Calculating results...\n\n")
            self.calculate_and_display_results()
            
        except Exception as e:
            self.results_text.insert(tk.END, f"Error during testing: {str(e)}\n")
        
        finally:
            # Reset UI
            self.test_btn.config(state='normal')
            self.stop_test_btn.config(state='disabled')
            self.testing_in_progress = False
            self.current_test_command = None
            
    def calculate_and_display_results(self):
        """Calculate accuracy and display results"""
        if not self.test_results:
            return
            
        # Calculate per-command accuracy
        commands = sorted(list(self.model_trainer.commands))
        accuracy_by_command = {}
        
        # Create confusion matrix
        self.confusion_matrix = np.zeros((len(commands), len(commands) + 1))
        cmd_to_idx = {cmd: i for i, cmd in enumerate(commands)}
        
        for cmd in commands:
            correct = 0
            total = len(self.test_results[cmd])
            
            if total == 0:
                accuracy_by_command[cmd] = 0
                continue
                
            for result in self.test_results[cmd]:
                if result == cmd:
                    correct += 1
                
                # Update confusion matrix
                cmd_idx = cmd_to_idx[cmd]
                if result is None:
                    # Last column is for unrecognized
                    self.confusion_matrix[cmd_idx, -1] += 1
                else:
                    result_idx = cmd_to_idx[result]
                    self.confusion_matrix[cmd_idx, result_idx] += 1
            
            accuracy_by_command[cmd] = (correct / total) * 100
        
        # Calculate overall accuracy
        all_results = [result for cmd_results in self.test_results.values() for result in cmd_results]
        all_expected = []
        for cmd, results in self.test_results.items():
            all_expected.extend([cmd] * len(results))
            
        correct = sum(1 for exp, res in zip(all_expected, all_results) if exp == res)
        overall_accuracy = (correct / len(all_results)) * 100 if all_results else 0
        
        # Display results
        self.results_text.insert(tk.END, f"Overall Accuracy: {overall_accuracy:.1f}%\n\n")
        self.results_text.insert(tk.END, "Accuracy by Command:\n")
        for cmd, acc in sorted(accuracy_by_command.items()):
            self.results_text.insert(tk.END, f"  {cmd}: {acc:.1f}%\n")
            
        # Display errors
        self.results_text.insert(tk.END, "\nCommand Recognition Errors:\n")
        for cmd in commands:
            error_counts = Counter()
            for result in self.test_results[cmd]:
                if result != cmd:
                    error_counts[result if result else "Not recognized"] += 1
                    
            if error_counts:
                self.results_text.insert(tk.END, f"  {cmd} → ")
                error_str = ", ".join(f"{error}: {count}" for error, count in error_counts.items())
                self.results_text.insert(tk.END, f"{error_str}\n")
                
        # Update confusion matrix visualization
        self.update_confusion_matrix()
        
    def update_confusion_matrix(self):
        """Update the confusion matrix visualization"""
        # Clear previous figure
        for widget in self.matrix_container.winfo_children():
            widget.destroy()
            
        if self.confusion_matrix is None:
            return
            
        # Create new figure
        plt.figure(figsize=(10, 8))
        fig = plt.Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)
        
        commands = sorted(list(self.model_trainer.commands))
        
        # Normalize by row
        row_sums = self.confusion_matrix.sum(axis=1)
        normalized_matrix = np.zeros_like(self.confusion_matrix, dtype=float)
        for i in range(len(row_sums)):
            if row_sums[i] > 0:
                normalized_matrix[i] = self.confusion_matrix[i] / row_sums[i]
                
        # Plot heat map
        cax = ax.matshow(normalized_matrix, cmap='Blues')
        fig.colorbar(cax)
        
        # Set ticks and labels
        ax.set_xticks(range(len(commands) + 1))
        ax.set_xticklabels([cmd[:6] for cmd in commands] + ['None'], rotation=45, ha='left')
        ax.set_yticks(range(len(commands)))
        ax.set_yticklabels(commands)
        
        # Add accuracy percentages to cells
        for i in range(len(commands)):
            for j in range(len(commands) + 1):
                if normalized_matrix[i, j] > 0:
                    ax.text(j, i, f"{normalized_matrix[i, j]:.2f}", ha="center", va="center", 
                            color="black" if normalized_matrix[i, j] < 0.5 else "white")
                            
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Recognized Command")
        ax.set_ylabel("Actual Command")
        fig.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.matrix_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def export_results(self):
        """Export test results to file"""
        if not self.test_results:
            messagebox.showinfo("Info", "No test results to export")
            return
            
        try:
            # Get file name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON Files", "*.json")],
                initialfile=f"accuracy_test_{timestamp}.json"
            )
            
            if not filename:
                return
                
            # Prepare export data
            export_data = {
                "timestamp": timestamp,
                "overall_accuracy": 0,
                "command_accuracy": {},
                "confusion_matrix": {}
            }
            
            # Calculate accuracy
            commands = sorted(list(self.model_trainer.commands))
            correct_by_command = {}
            total_by_command = {}
            
            for cmd in commands:
                correct = 0
                total = len(self.test_results[cmd])
                
                if total == 0:
                    continue
                    
                for result in self.test_results[cmd]:
                    if result == cmd:
                        correct += 1
                        
                correct_by_command[cmd] = correct
                total_by_command[cmd] = total
                export_data["command_accuracy"][cmd] = (correct / total) * 100
            
            # Calculate overall accuracy
            total_correct = sum(correct_by_command.values())
            total_tests = sum(total_by_command.values())
            export_data["overall_accuracy"] = (total_correct / total_tests) * 100 if total_tests else 0
            
            # Add confusion matrix
            if self.confusion_matrix is not None:
                cmd_to_idx = {cmd: i for i, cmd in enumerate(commands)}
                for cmd in commands:
                    cmd_idx = cmd_to_idx[cmd]
                    confusion_row = {}
                    
                    for result_cmd in commands:
                        result_idx = cmd_to_idx[result_cmd]
                        if self.confusion_matrix[cmd_idx, result_idx] > 0:
                            confusion_row[result_cmd] = int(self.confusion_matrix[cmd_idx, result_idx])
                            
                    if self.confusion_matrix[cmd_idx, -1] > 0:
                        confusion_row["not_recognized"] = int(self.confusion_matrix[cmd_idx, -1])
                        
                    export_data["confusion_matrix"][cmd] = confusion_row
            
            # Add raw results
            export_data["raw_results"] = {cmd: results for cmd, results in self.test_results.items()}
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            messagebox.showinfo("Success", f"Results exported to {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting results: {str(e)}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    MinimalRunInterface().run()
