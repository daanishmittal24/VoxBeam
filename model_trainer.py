import torch
import whisper
import numpy as np
import os
from pathlib import Path
import json
from tqdm import tqdm
from datetime import datetime
from audio_augmenter import AudioAugmenter, augment_command_data

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
        
        # Create augmenter
        self.augmenter = AudioAugmenter()

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

    def train_model(self, command_folders, epochs=30, batch_size=4, learning_rate=1e-4, 
                   use_augmentation=True, augmentations_per_sample=3):
        """Update command vocabulary from command folders and generate augmented data if needed"""
        try:
            # Load model if not loaded
            if self.model is None:
                if not self.load_base_model():
                    return False

            # Extract command names
            self.commands = {folder.name.replace('_', ' ').lower() for folder in command_folders}
            
            # Generate synthetic data with augmentation if enabled
            if use_augmentation:
                self.current_operation = "Generating augmented data..."
                self.training_progress = 0
                
                # Create augmented data directory
                augmented_dir = self.model_save_dir / "augmented_data"
                
                # Progress callback function to update training progress
                def progress_callback(progress, operation_info):
                    self.training_progress = progress
                    self.current_operation = operation_info
                
                # Generate augmented data
                augmented_files = augment_command_data(
                    command_data_dir=str(command_folders[0].parent),
                    output_dir=str(augmented_dir),
                    augmentations_per_sample=augmentations_per_sample,
                    progress_callback=progress_callback
                )
                
                # Log augmentation results
                total_augmented_files = sum(len(files) for files in augmented_files.values())
                print(f"Generated {total_augmented_files} augmented audio files")
            
            # Save commands list
            commands_file = self.model_save_dir / "commands.json"
            with open(commands_file, "w") as f:
                json.dump(list(self.commands), f)

            self.current_operation = "Training completed"
            self.training_progress = 100
            return True

        except Exception as e:
            print(f"Error in training: {e}")
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
            if (audio_data.dtype != np.float32):
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio volume
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Create a prompt with all available commands
            commands_list = ", ".join(sorted(self.commands))
            initial_prompt = f"VoxBeam. Available commands are: {commands_list}. The spoken command is:"
            
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

    def evaluate_model_accuracy(self, test_data_dir=None, save_results=True):
        """Evaluate model accuracy on test data"""
        try:
            if self.model is None:
                if not self.load_trained_model():
                    return None
                    
            if test_data_dir is None:
                test_data_dir = Path("command_data")
                
            # Get all command folders
            command_dirs = [d for d in test_data_dir.iterdir() if d.is_dir()]
            
            results = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "command_accuracy": {},
                "confusion_matrix": {},
                "raw_results": {},
            }
            
            total_correct = 0
            total_samples = 0
            
            self.current_operation = "Evaluating model accuracy..."
            self.training_progress = 0
            
            for command_dir in command_dirs:
                command_name = command_dir.name.replace('_', ' ').lower()
                audio_files = list(command_dir.glob("*.wav"))
                
                if not audio_files:
                    continue
                    
                correct = 0
                results["confusion_matrix"][command_name] = {}
                results["raw_results"][command_name] = []
                
                for i, audio_file in enumerate(audio_files):
                    # Update progress
                    progress = (total_samples / (len(command_dirs) * len(audio_files))) * 100
                    self.training_progress = progress
                    self.current_operation = f"Testing {command_name} ({i+1}/{len(audio_files)})"
                    
                    # Load and transcribe audio
                    audio_data = self.load_audio_file(audio_file)
                    if audio_data is None:
                        continue
                        
                    transcription = self.transcribe_audio(audio_data)
                    results["raw_results"][command_name].append(transcription)
                    
                    # Track results
                    if transcription == command_name:
                        correct += 1
                        total_correct += 1
                    
                    # Update confusion matrix
                    if transcription:
                        if transcription not in results["confusion_matrix"][command_name]:
                            results["confusion_matrix"][command_name][transcription] = 0
                        results["confusion_matrix"][command_name][transcription] += 1
                    else:
                        # Handle unrecognized commands
                        if "not_recognized" not in results["confusion_matrix"][command_name]:
                            results["confusion_matrix"][command_name]["not_recognized"] = 0
                        results["confusion_matrix"][command_name]["not_recognized"] += 1
                    
                    total_samples += 1
                    
                # Calculate accuracy for this command
                accuracy = (correct / len(audio_files)) * 100
                results["command_accuracy"][command_name] = accuracy
            
            # Calculate overall accuracy
            if total_samples > 0:
                results["overall_accuracy"] = (total_correct / total_samples) * 100
            else:
                results["overall_accuracy"] = 0
            
            # Save results
            if save_results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = Path(f"accuracy_test_{timestamp}.json")
                with open(results_file, "w") as f:
                    json.dump(results, f, indent=2)
                    
                print(f"Saved accuracy results to {results_file}")
            
            self.current_operation = "Evaluation complete"
            self.training_progress = 100
            return results
            
        except Exception as e:
            print(f"Error evaluating model: {e}")
            self.current_operation = f"Error: {str(e)}"
            return None
            
    def load_audio_file(self, file_path):
        """Load audio file for evaluation"""
        try:
            return self.augmenter.load_audio(file_path)
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None

    def get_progress(self):
        return {
            "progress": self.training_progress,
            "operation": self.current_operation
        }
