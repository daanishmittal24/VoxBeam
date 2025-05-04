import tkinter as tk
from tkinter import ttk, messagebox
import threading
import librosa
from model_trainer import ModelTrainer
from command_organizer import CommandOrganizer
from command_executor import CommandExecutor
from audio_recorder import AudioRecorder

class MinimalRunInterface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Voice Command Minimal Runner")
        self.root.geometry("400x350")

        self.model_trainer = ModelTrainer()
        self.command_organizer = CommandOrganizer()
        self.command_executor = CommandExecutor()
        self.audio_recorder = AudioRecorder()
        self.listening = False
        self.listen_thread = None

        self.setup_gui()
        self.load_commands()

    def setup_gui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready", font=("Arial", 12))
        self.status_label.pack(pady=5)

        # Listen controls
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=5)
        self.start_btn = ttk.Button(btn_frame, text="Start Listening", command=self.start_listening)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="Stop Listening", command=self.stop_listening, state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Manual command buttons
        self.cmds_frame = ttk.LabelFrame(main_frame, text="Manual Commands")
        self.cmds_frame.pack(fill=tk.BOTH, expand=True, pady=10)

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

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    MinimalRunInterface().run()
