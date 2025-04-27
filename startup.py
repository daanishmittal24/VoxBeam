import tkinter as tk
from tkinter import ttk
from model_trainer import ModelTrainer
from command_organizer import CommandOrganizer
from gui_modes import RecordingMode, TrainingMode, DeploymentMode

class StartupWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Voice Command System")
        self.root.geometry("500x400")
        
        self.model_trainer = ModelTrainer()
        self.command_organizer = CommandOrganizer()
        
        # Create custom styles
        self.setup_styles()
        self.setup_gui()
    
    def setup_styles(self):
        style = ttk.Style()
        style.configure('Record.TButton', padding=10)
        style.configure('Recording.TButton', padding=10, background='red')
        style.configure('Mode.TButton', padding=15, font=('Arial', 11))
        style.configure('Action.TButton', padding=10, font=('Arial', 10, 'bold'))
    
    def setup_gui(self):
        # Mode selection frame
        mode_frame = ttk.LabelFrame(self.root, text="Select Mode")
        mode_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Record mode button
        record_btn = ttk.Button(
            mode_frame,
            text="Recording Mode\nRecord voice commands",
            style='Mode.TButton',
            command=self.start_recording_mode
        )
        record_btn.pack(pady=10, padx=20, fill=tk.X)

        # Train mode button
        train_btn = ttk.Button(
            mode_frame,
            text="Training Mode\nTrain the voice command model",
            style='Mode.TButton',
            command=self.start_training_mode
        )
        train_btn.pack(pady=10, padx=20, fill=tk.X)

        # Deploy/test mode button
        deploy_btn = ttk.Button(
            mode_frame,
            text="Testing Mode\nTest trained voice commands",
            style='Mode.TButton',
            command=self.start_deployment_mode
        )
        deploy_btn.pack(pady=10, padx=20, fill=tk.X)

    def start_recording_mode(self):
        window = tk.Toplevel(self.root)
        RecordingMode(window, self.command_organizer)
        window.protocol("WM_DELETE_WINDOW", lambda: self.close_mode_window(window))

    def start_training_mode(self):
        window = tk.Toplevel(self.root)
        TrainingMode(window, self.model_trainer, self.command_organizer)
        window.protocol("WM_DELETE_WINDOW", lambda: self.close_mode_window(window))

    def start_deployment_mode(self):
        # Check if model is trained first
        if not self.model_trainer.load_trained_model():
            tk.messagebox.showwarning(
                "No Trained Model", 
                "No trained model found. Please train the model first."
            )
            return
            
        window = tk.Toplevel(self.root)
        DeploymentMode(window, self.model_trainer, self.command_organizer)
        window.protocol("WM_DELETE_WINDOW", lambda: self.close_mode_window(window))

    def close_mode_window(self, window):
        window.destroy()
        # Re-enable the startup window if all mode windows are closed
        if not any(isinstance(w, tk.Toplevel) for w in self.root.winfo_children()):
            self.root.deiconify()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = StartupWindow()
    app.run()