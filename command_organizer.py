import shutil
from pathlib import Path
import json

class CommandOrganizer:
    def __init__(self, base_dir="command_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.commands_file = self.base_dir / "commands.json"
        self.commands = self.load_commands()

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