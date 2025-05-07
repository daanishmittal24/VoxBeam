import shutil
from pathlib import Path
import json
import os

class CommandOrganizer:
    def __init__(self, base_dir="command_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.commands_file = self.base_dir / "commands.json"
        self.commands = self.load_commands()
        
        # For bulk training
        self.bulk_training_mode = False
        self.current_bulk_command = None
        self.bulk_recordings_processed = 0

    def load_commands(self):
        """Load command mappings from JSON file"""
        if self.commands_file.exists():
            try:
                with open(self.commands_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_commands(self):
        """Save command mappings to JSON file"""
        with open(self.commands_file, 'w') as f:
            json.dump(self.commands, f, indent=2)

    def add_recording(self, recording_path, command_text):
        """Add a recording to the appropriate command folder"""
        # Create safe folder name from command
        folder_name = command_text.lower().replace(" ", "_")
        command_dir = self.base_dir / folder_name
        command_dir.mkdir(exist_ok=True)

        # Copy recording to command folder
        recording_path = Path(recording_path)
        new_path = command_dir / recording_path.name
        shutil.copy2(recording_path, new_path)

        # Update commands mapping
        self.commands[str(new_path)] = command_text
        self.save_commands()

        return str(new_path)

    def get_command_folders(self):
        """Get list of all command folders"""
        return [d for d in self.base_dir.iterdir() if d.is_dir()]

    def get_recordings_for_command(self, command_text):
        """Get all recordings for a specific command"""
        folder_name = command_text.lower().replace(" ", "_")
        command_dir = self.base_dir / folder_name
        if command_dir.exists():
            return list(command_dir.glob("*.wav"))
        return []

    def get_all_commands(self):
        """Get list of all unique commands"""
        return list(set(self.commands.values()))
        
    def start_bulk_training(self, command_text):
        """Start bulk training mode for a specific command"""
        self.bulk_training_mode = True
        self.current_bulk_command = command_text
        self.bulk_recordings_processed = 0
        
        # Create the command directory if it doesn't exist
        folder_name = command_text.lower().replace(" ", "_")
        command_dir = self.base_dir / folder_name
        command_dir.mkdir(exist_ok=True)
        
        return True
        
    def stop_bulk_training(self):
        """Stop bulk training mode"""
        result = {
            "command": self.current_bulk_command,
            "recordings_processed": self.bulk_recordings_processed
        }
        
        self.bulk_training_mode = False
        self.current_bulk_command = None
        
        return result
        
    def process_recording_for_bulk_training(self, recording_path):
        """Process a recording for the currently active bulk training command"""
        if not self.bulk_training_mode or not self.current_bulk_command:
            return False
            
        result = self.add_recording(recording_path, self.current_bulk_command)
        self.bulk_recordings_processed += 1
        return result
        
    def is_bulk_training_active(self):
        """Check if bulk training mode is active"""
        return self.bulk_training_mode
        
    def get_bulk_training_status(self):
        """Get current status of bulk training"""
        return {
            "active": self.bulk_training_mode,
            "command": self.current_bulk_command,
            "recordings_processed": self.bulk_recordings_processed
        }
        
    def create_new_command(self, command_text):
        """Create a new command folder without adding recordings"""
        folder_name = command_text.lower().replace(" ", "_")
        command_dir = self.base_dir / folder_name
        command_dir.mkdir(exist_ok=True)
        return str(command_dir)
        
    def bulk_import_recordings(self, source_folder, command_text):
        """Import all recordings from a folder into a command folder"""
        source_folder = Path(source_folder)
        if not source_folder.exists() or not source_folder.is_dir():
            return {"success": False, "error": "Source folder does not exist"}
            
        folder_name = command_text.lower().replace(" ", "_")
        command_dir = self.base_dir / folder_name
        command_dir.mkdir(exist_ok=True)
        
        count = 0
        for file in source_folder.glob("*.wav"):
            try:
                new_path = command_dir / file.name
                shutil.copy2(file, new_path)
                self.commands[str(new_path)] = command_text
                count += 1
            except Exception as e:
                print(f"Error copying file {file}: {e}")
                
        self.save_commands()
        return {"success": True, "count": count, "command": command_text}