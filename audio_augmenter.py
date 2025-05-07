import numpy as np
import soundfile as sf
import librosa
import random
import os
from pathlib import Path
from scipy import signal
import json
from tqdm import tqdm
import wave
from datetime import datetime

class AudioAugmenter:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.augmentation_types = [
            'pitch_shift',
            'time_stretch',
            'volume_adjust',
            'add_noise',
            'shift',
            'reverb'
        ]
    
    def load_audio(self, file_path):
        """Load audio file and convert to the correct format"""
        try:
            audio_data, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            return audio_data
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None
    
    def save_audio(self, audio_data, output_path):
        """Save augmented audio to file"""
        try:
            # Normalize
            audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-6)
            
            # Convert float32 to int16 for wave file
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            with wave.open(str(output_path), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
                
            return True
        except Exception as e:
            print(f"Error saving audio file {output_path}: {e}")
            return False
    
    def pitch_shift(self, audio_data, n_steps=None):
        """Shift the pitch of the audio by n_steps semitones"""
        if n_steps is None:
            n_steps = random.uniform(-3.0, 3.0)
        return librosa.effects.pitch_shift(audio_data, sr=self.sample_rate, n_steps=n_steps)
    
    def time_stretch(self, audio_data, rate=None):
        """Stretch or compress the audio by rate"""
        if rate is None:
            rate = random.uniform(0.8, 1.2)
        return librosa.effects.time_stretch(audio_data, rate=rate)
    
    def volume_adjust(self, audio_data, factor=None):
        """Adjust the volume by factor"""
        if factor is None:
            factor = random.uniform(0.6, 1.5)
        return audio_data * factor
    
    def add_noise(self, audio_data, noise_level=None):
        """Add white noise to the audio"""
        if noise_level is None:
            noise_level = random.uniform(0.001, 0.01)
        noise = np.random.normal(0, noise_level, len(audio_data))
        return audio_data + noise
    
    def shift(self, audio_data, shift_percent=None):
        """Shift the audio in time"""
        if shift_percent is None:
            shift_percent = random.uniform(-0.2, 0.2)
        
        shift_amount = int(len(audio_data) * shift_percent)
        if shift_amount > 0:
            # Shift right
            result = np.zeros_like(audio_data)
            result[shift_amount:] = audio_data[:-shift_amount]
            return result
        elif shift_amount < 0:
            # Shift left
            result = np.zeros_like(audio_data)
            result[:shift_amount] = audio_data[-shift_amount:]
            return result
        else:
            return audio_data
    
    def reverb(self, audio_data, reverb_level=None):
        """Add reverb effect to audio"""
        if reverb_level is None:
            reverb_level = random.uniform(0.1, 0.3)
        
        reverb_length = int(self.sample_rate * reverb_level)
        decay = np.exp(-np.linspace(0, 5, reverb_length))
        impulse_response = np.random.randn(reverb_length) * decay
        
        # Convolve with impulse response
        reverbed_audio = signal.convolve(audio_data, impulse_response, mode='full')
        
        # Trim to original length
        reverbed_audio = reverbed_audio[:len(audio_data)]
        
        # Normalize
        reverbed_audio = reverbed_audio / (np.max(np.abs(reverbed_audio)) + 1e-6)
        
        return reverbed_audio
    
    def apply_random_augmentation(self, audio_data):
        """Apply a random augmentation to the audio"""
        aug_type = random.choice(self.augmentation_types)
        
        if aug_type == 'pitch_shift':
            return self.pitch_shift(audio_data)
        elif aug_type == 'time_stretch':
            return self.time_stretch(audio_data)
        elif aug_type == 'volume_adjust':
            return self.volume_adjust(audio_data)
        elif aug_type == 'add_noise':
            return self.add_noise(audio_data)
        elif aug_type == 'shift':
            return self.shift(audio_data)
        elif aug_type == 'reverb':
            return self.reverb(audio_data)
        
        return audio_data
    
    def apply_combined_augmentation(self, audio_data, num_augs=2):
        """Apply multiple augmentations to the audio"""
        augmented_audio = audio_data.copy()
        
        # Select a random number of augmentations to apply (1 to num_augs)
        num_to_apply = random.randint(1, num_augs)
        aug_types = random.sample(self.augmentation_types, num_to_apply)
        
        for aug_type in aug_types:
            if aug_type == 'pitch_shift':
                augmented_audio = self.pitch_shift(augmented_audio)
            elif aug_type == 'time_stretch':
                augmented_audio = self.time_stretch(augmented_audio)
            elif aug_type == 'volume_adjust':
                augmented_audio = self.volume_adjust(augmented_audio)
            elif aug_type == 'add_noise':
                augmented_audio = self.add_noise(augmented_audio)
            elif aug_type == 'shift':
                augmented_audio = self.shift(augmented_audio)
            elif aug_type == 'reverb':
                augmented_audio = self.reverb(augmented_audio)
        
        return augmented_audio
    
    def generate_augmented_dataset(self, command_folders, output_base_dir, augmentations_per_sample=3, progress_callback=None):
        """Generate augmented dataset from command folders"""
        augmented_files = {}
        
        for command_folder in command_folders:
            command_name = command_folder.name
            output_dir = Path(output_base_dir) / command_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # List all audio files in the command folder
            audio_files = list(command_folder.glob("*.wav"))
            total_files = len(audio_files) * augmentations_per_sample
            
            augmented_files[command_name] = []
            
            # Generate augmentations for each audio file
            file_count = 0
            for audio_file in audio_files:
                audio_data = self.load_audio(audio_file)
                if audio_data is None:
                    continue
                
                # Create augmentations
                for i in range(augmentations_per_sample):
                    augmented_audio = self.apply_combined_augmentation(audio_data)
                    
                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    output_path = output_dir / f"aug_{timestamp}_{i}.wav"
                    
                    if self.save_audio(augmented_audio, output_path):
                        augmented_files[command_name].append(str(output_path))
                    
                    file_count += 1
                    if progress_callback:
                        progress_percentage = (file_count / total_files) * 100
                        progress_callback(progress_percentage, f"Augmenting {command_name} files")
        
        return augmented_files


def augment_command_data(command_data_dir="command_data", output_dir="augmented_data", 
                         augmentations_per_sample=3, progress_callback=None):
    """Main function to augment all command data"""
    # Initialize augmenter
    augmenter = AudioAugmenter()
    
    # Find all command folders
    command_dir = Path(command_data_dir)
    command_folders = [d for d in command_dir.iterdir() if d.is_dir()]
    
    # Create output directory
    output_base_dir = Path(output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate augmented dataset
    augmented_files = augmenter.generate_augmented_dataset(
        command_folders, 
        output_base_dir, 
        augmentations_per_sample=augmentations_per_sample,
        progress_callback=progress_callback
    )
    
    # Save metadata
    metadata_path = output_base_dir / "augmentation_metadata.json"
    with open(metadata_path, 'w') as f:
        metadata = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "augmentations_per_sample": augmentations_per_sample,
            "original_data_dir": str(command_data_dir),
            "augmented_files_count": {cmd: len(files) for cmd, files in augmented_files.items()},
            "total_augmented_files": sum(len(files) for files in augmented_files.values())
        }
        json.dump(metadata, f, indent=2)
    
    return augmented_files