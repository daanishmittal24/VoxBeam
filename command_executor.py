import subprocess
import os
from pynput.keyboard import Key, Controller

class CommandExecutor:
    def __init__(self):
        self.keyboard = Controller()
        self.commands = {
            "open notepad": self._open_notepad,
            "open calculator": self._open_calculator,
            "close window": self._close_window
        }
        print(f"Available commands: {list(self.commands.keys())}")

    def execute_command(self, command_text):
        """Execute the corresponding command based on the recognized text."""
        command_text = command_text.lower().strip()
        print(f"Attempting to execute command: '{command_text}'")
        
        if command_text in self.commands:
            print(f"Found matching command: '{command_text}'")
            success = self.commands[command_text]()
            print(f"Command execution {'succeeded' if success else 'failed'}")
            return success
        else:
            print(f"No matching command found for: '{command_text}'")
            return False

    def _open_notepad(self):
        """Open Windows Notepad."""
        try:
            subprocess.Popen('notepad.exe')
            return True
        except Exception as e:
            print(f"Error opening Notepad: {e}")
            return False

    def _open_calculator(self):
        """Open Windows Calculator."""
        try:
            subprocess.Popen('calc.exe')
            return True
        except Exception as e:
            print(f"Error opening Calculator: {e}")
            return False

    def _close_window(self):
        """Simulate Alt+F4 to close the active window."""
        try:
            with self.keyboard.pressed(Key.alt):
                self.keyboard.press(Key.f4)
                self.keyboard.release(Key.f4)
            return True
        except Exception as e:
            print(f"Error closing window: {e}")
            return False