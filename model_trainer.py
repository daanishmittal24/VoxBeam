import torch
import whisper
import numpy as np
import os
from pathlib import Path
import json
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, model_save_dir="trained_models"):
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)
        
        self.device = self._setup_device()
        self.model = None
        self.training_progress = 0
        self.current_operation = "Idle"
        
        # Store available commands
        self.commands = set()

    def _setup_device(self):
        if not torch.cuda.is_available():
            print("Warning: CUDA is not available, using CPU instead")
            return torch.device("cpu")
        try:
            cuda_device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            return cuda_device
        except Exception as e:
            print(f"Error initializing CUDA: {e}")
            print("Falling back to CPU")
            return torch.device("cpu")

    def load_base_model(self):
        """Load the base Whisper model"""
        self.current_operation = "Loading base model..."
        try:
            # Use the 'base' model for quick command recognition
            self.model = whisper.load_model("base")
            self.current_operation = "Base model loaded successfully"
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.current_operation = f"Error loading model: {str(e)}"
            return False

    def train_model(self, command_folders, epochs=30, batch_size=4, learning_rate=1e-4):
        """Update command vocabulary from command folders"""
        try:
            # Load model if not loaded
            if self.model is None:
                if not self.load_base_model():
                    return False

            # Extract command names
            self.commands = {folder.name.replace('_', ' ').lower() for folder in command_folders}
            
            # Save commands list
            commands_file = self.model_save_dir / "commands.json"
            with open(commands_file, "w") as f:
                json.dump(list(self.commands), f)

            self.current_operation = "Commands updated"
            self.training_progress = 100
            return True

        except Exception as e:
            print(f"Error updating commands: {e}")
            self.current_operation = f"Error: {str(e)}"
            return False

    def load_trained_model(self, model_path=None):
        """Load the base model and command list"""
        try:
            if model_path is None:
                model_path = self.model_save_dir

            # Load base model
            if not self.load_base_model():
                return False

            # Load commands list
            commands_file = Path(model_path) / "commands.json"
            if commands_file.exists():
                with open(commands_file) as f:
                    self.commands = set(json.load(f))
                return True
            else:
                self.current_operation = "No commands list found"
                return False

        except Exception as e:
            self.current_operation = f"Error loading model: {str(e)}"
            return False

    def transcribe_audio(self, audio_data, sample_rate=16000):
        """Transcribe audio using Whisper with optimizations for command recognition"""
        try:
            if self.model is None:
                if not self.load_base_model():
                    raise Exception("Failed to load model")

            # Convert audio to float32 if needed
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio volume
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Create a prompt with all available commands
            commands_list = ", ".join(sorted(self.commands))
            initial_prompt = f"Voice command system. Available commands are: {commands_list}. The spoken command is:"
            
            # Perform transcription with specific options for command recognition
            result = self.model.transcribe(
                audio_data,
                language="en",
                initial_prompt=initial_prompt,
                task="transcribe",
                beam_size=5,
                best_of=5,
                temperature=0.0,  # Use greedy decoding for more consistent results
                fp16=torch.cuda.is_available()
            )
            
            transcription = result["text"].lower().strip().strip('.')
            print(f"Raw transcription: '{transcription}'")

            # 1. First try exact match (case insensitive)
            for cmd in self.commands:
                if transcription == cmd.lower():
                    print(f"Found exact match: '{cmd}'")
                    return cmd

            # 2. Try matching key words only (e.g., "notepad" matches "open notepad")
            # Get all unique words from commands
            cmd_words = {}
            for cmd in self.commands:
                for word in cmd.lower().split():
                    if word not in ['open', 'close', 'the', 'a', 'an']:  # Skip common verbs and articles
                        if word not in cmd_words:
                            cmd_words[word] = []
                        cmd_words[word].append(cmd)

            # Check if transcription contains any command words
            trans_words = transcription.split()
            matched_commands = {}
            
            for word in trans_words:
                if word in cmd_words:
                    for cmd in cmd_words[word]:
                        if cmd not in matched_commands:
                            matched_commands[cmd] = 0
                        matched_commands[cmd] += 1
            
            # Find best match based on number of matching keywords
            if matched_commands:
                best_cmd = max(matched_commands.items(), key=lambda x: x[1])[0]
                print(f"Found keyword match: '{best_cmd}' (matched {matched_commands[best_cmd]} keywords)")
                return best_cmd

            # 3. Try fuzzy matching as a fallback
            best_match = None
            best_score = 0
            
            for cmd in self.commands:
                # Split commands and transcription into words
                cmd_parts = cmd.lower().split()
                trans_parts = transcription.split()
                
                # Score based on partial matches
                score = 0
                for trans_word in trans_parts:
                    for cmd_word in cmd_parts:
                        if trans_word in cmd_word or cmd_word in trans_word:
                            score += 1 if trans_word == cmd_word else 0.5
                
                # Normalize score by command length
                score = score / len(cmd_parts)
                print(f"Command '{cmd}' fuzzy score: {score:.2f}")
                
                if score > best_score and score >= 0.5:  # Threshold of 0.5
                    best_match = cmd
                    best_score = score

            if best_match:
                print(f"Found fuzzy match: '{best_match}' (score: {best_score:.2f})")
                return best_match
            
            print("No command match found")
            return None

        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    def get_progress(self):
        return {
            "progress": self.training_progress,
            "operation": self.current_operation
        }
