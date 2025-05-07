import torch
import whisper
import numpy as np
import os
from pathlib import Path
import json
from tqdm import tqdm
import string
from difflib import SequenceMatcher

# Add imports for online service integration
try:
    from online_transcription import OnlineTranscriptionService
    online_available = True
except ImportError:
    online_available = False

class ModelTrainer:
    def __init__(self, model_save_dir="trained_models"):
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)
        
        self.device = self._setup_device()
        self.model = None
        self.model_size = "base"  # Use "base" model for better balance
        self.training_progress = 0
        self.current_operation = "Idle"
        
        # Store available commands
        self.commands = set()
        
        # Common words to filter out and confidence thresholds
        self.filter_words = ["um", "uh", "eh", "ah", "like", "so", "you know", "actually", "basically", "literally"]
        self.confidence_threshold = 0.60  # Similarity threshold for command matching
        
        # Memory management - explicitly define the attribute
        self.optimize_memory = True
        
        # Online service integration
        self.use_online_service = True  # Set to True to prioritize online services
        if online_available:
            self.online_service = OnlineTranscriptionService()
        else:
            self.online_service = None
            self.use_online_service = False
            
        print(f"Online transcription services {'available' if self.use_online_service else 'not available'}")

    def _setup_device(self):
        if not torch.cuda.is_available():
            print("Warning: CUDA is not available, using CPU instead")
            return torch.device("cpu")
        try:
            # Memory optimization for limited VRAM
            torch.cuda.set_per_process_memory_fraction(0.7)  # Use only 70% of available VRAM
            torch.cuda.empty_cache()  # Clear CUDA cache
            
            cuda_device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"Memory optimization enabled: {self.optimize_memory}")
            return cuda_device
        except Exception as e:
            print(f"Error initializing CUDA: {e}")
            print("Falling back to CPU")
            return torch.device("cpu")

    def load_base_model(self):
        """Load the Whisper model"""
        self.current_operation = f"Loading {self.model_size} model..."
        try:
            # First clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Load the model with optimized settings
            self.model = whisper.load_model(
                self.model_size,
                device=self.device,
                download_root=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            )
            
            # Low precision to reduce memory usage
            if self.optimize_memory and self.device.type == "cuda":
                self.model = self.model.half()  # Convert to half precision
                
            self.current_operation = f"{self.model_size.capitalize()} model loaded successfully"
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.current_operation = f"Error loading model: {str(e)}"
            # Try to fall back to an even smaller model if available
            try:
                print("Attempting to fall back to tiny model...")
                self.model_size = "tiny"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.model = whisper.load_model("tiny")
                self.model = self.model.half() if self.optimize_memory and self.device.type == "cuda" else self.model
                self.current_operation = "Tiny model loaded successfully (fallback)"
                return True
            except:
                return False

    def train_model(self, command_folders, epochs=5, batch_size=4, learning_rate=1e-4):
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

    def _similarity_score(self, a, b):
        """Calculate similarity between two strings using sequence matcher"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _normalize_text(self, text):
        """Clean and normalize text to improve matching"""
        if not text:
            return ""
            
        # Convert to lowercase and strip punctuation
        text = text.lower().strip()
        text = ''.join(c for c in text if c not in string.punctuation)
        
        # Remove filter words
        for word in self.filter_words:
            text = text.replace(f" {word} ", " ")
        
        # Remove extra whitespaces
        text = " ".join(text.split())
        
        return text

    def transcribe_audio(self, audio_data, sample_rate=16000, audio_file=None):
        """Transcribe audio using online service or local Whisper model
        
        Args:
            audio_data: Numpy array of audio data
            sample_rate: Audio sample rate
            audio_file: Optional path to audio file (for online services)
        """
        # Try online service first if enabled and audio_file is provided
        if self.use_online_service and self.online_service and audio_file:
            try:
                print("Attempting transcription with online service...")
                transcription = self.online_service.transcribe_file(audio_file)
                if transcription:
                    print(f"Online transcription successful: '{transcription}'")
                    
                    # Process the result to find matching command
                    normalized_text = self._normalize_text(transcription)
                    return self._match_command_to_text(normalized_text, transcription)
            except Exception as e:
                print(f"Online transcription failed: {e}, falling back to local model")
        
        # Fall back to local Whisper model
        try:
            if self.model is None:
                if not self.load_base_model():
                    raise Exception("Failed to load model")
                    
            # Clear CUDA cache before processing
            if self.optimize_memory and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Convert audio to float32 if needed
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio volume
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Regular command recognition path
            # Create a prompt with all available commands
            commands_list = ", ".join(sorted(self.commands))
            initial_prompt = f"Voice command system. Available commands are: {commands_list}. The spoken command is:"
            
            # Perform transcription with optimized resource settings
            result = self.model.transcribe(
                audio_data,
                language="en",
                initial_prompt=initial_prompt,
                task="transcribe",
                beam_size=1,     # Resource-efficient setting
                best_of=1,       
                temperature=0.0,  # Use greedy decoding for more consistent results
                fp16=self.optimize_memory and torch.cuda.is_available()
            )
            
            # Extract and normalize transcription
            raw_transcription = result["text"]
            transcription = self._normalize_text(raw_transcription)
            print(f"Raw transcription: '{raw_transcription}'")
            print(f"Normalized transcription: '{transcription}'")
            
            return self._match_command_to_text(transcription, raw_transcription)

        except Exception as e:
            print(f"Transcription error: {e}")
            # Attempt to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
            
    def _match_command_to_text(self, normalized_text, raw_text=None):
        """Match normalized text to available commands"""
        transcription = normalized_text
        
        # Prioritize similarity matching over exact matching for better results
        best_match = None
        best_score = 0
        
        # Try similarity matching against all commands
        for cmd in self.commands:
            cmd_normalized = self._normalize_text(cmd)
            score = self._similarity_score(transcription, cmd_normalized)
            print(f"Command '{cmd}' similarity score: {score:.3f}")
            
            if score > best_score:
                best_match = cmd
                best_score = score
        
        if best_score >= self.confidence_threshold:
            print(f"Found similarity match: '{best_match}' (score: {best_score:.3f})")
            return best_match
        
        # Split transcription into words and check for exact word matches
        trans_words = set(transcription.split())
        
        # Check if the transcription contains multiple commands
        detected_commands = []
        
        # First try exact match (highest priority)
        for cmd in self.commands:
            cmd_normalized = self._normalize_text(cmd)
            if cmd_normalized == transcription:
                detected_commands = [cmd]  # If exact match, use only this command
                print(f"Found exact match: '{cmd}'")
                return detected_commands[0]  # Return immediately on exact match
                
        # If no exact match, try more rigorous detection 
        if not detected_commands:
            # Split transcription into words for word-by-word matching
            trans_words = transcription.split()
            
            # Higher confidence threshold for multi-command detection to prevent false positives
            multi_cmd_threshold = 0.75
            
            for cmd in sorted(self.commands, key=len, reverse=True):
                cmd_normalized = self._normalize_text(cmd)
                
                # For each command, compute a proper similarity score
                # This prevents partial matches like "a" in "attack" matching with "a" in "back"
                score = self._similarity_score(transcription, cmd_normalized)
                
                # Calculate word-based match score (stricter than substring match)
                cmd_words = set(cmd_normalized.split())
                word_match_ratio = len(cmd_words.intersection(trans_words)) / len(cmd_words) if cmd_words else 0
                
                # Only add commands with high similarity or complete word matches
                if score >= multi_cmd_threshold or (word_match_ratio == 1.0 and len(cmd_words) > 0):
                    # Additional check: for single word commands, ensure it's a complete word match
                    if len(cmd_words) == 1:
                        cmd_word = next(iter(cmd_words))
                        # Check if it appears as a complete word (not part of another word)
                        if cmd_word in trans_words:
                            detected_commands.append(cmd)
                    else:
                        # For multi-word commands, the word_match_ratio check above is sufficient
                        if word_match_ratio == 1.0:
                            detected_commands.append(cmd)
                        
        # If we found multiple commands, return them all
        if len(detected_commands) > 1:
            print(f"Multiple commands detected: {detected_commands}")
            return detected_commands
        
        # If we found exactly one command
        if len(detected_commands) == 1:
            print(f"Found match: '{detected_commands[0]}'")
            return detected_commands[0]

        # Try keyword matching as fallback
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
            keyword_score = matched_commands[best_cmd] / len(best_cmd.split())
            if keyword_score >= 0.5:  # At least half the keywords match
                print(f"Found keyword match: '{best_cmd}' (matched {matched_commands[best_cmd]} keywords)")
                return best_cmd
        
        print("No command match found")
        return None
            
    def cleanup(self):
        """Clean up resources to free memory"""
        try:
            # Delete model and free memory
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Cleaned up model resources")
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def get_progress(self):
        return {
            "progress": self.training_progress,
            "operation": self.current_operation
        }
