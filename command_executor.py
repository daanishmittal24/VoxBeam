import subprocess
import os
from pynput.keyboard import Key, Controller
from pynput.mouse import Button, Controller as MouseController
import webbrowser
from time import sleep

class CommandExecutor:
    def __init__(self):
        self.keyboard = Controller()
        self.mouse = MouseController()
        self.commands = {
            "open notepad": self._open_notepad,
            "open calculator": self._open_calculator,
            "close window": self._close_window,
            "click": self._click,
            "scroll up": self._scroll_up,
            "scroll down": self._scroll_down,
            "open google search": self._open_google_search,
            "attack": self._attack,
            "back": self._back,
            "forward": self._forward,
            "left": self._left,
            "right": self._right
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
            # with self.keyboard.pressed(Key.alt):
            #     self.keyboard.press(Key.f4)
            #     self.keyboard.release(Key.f4)
            return True
        except Exception as e:
            print(f"Error closing window: {e}")
            return False

    def _click(self):
        """Simulate a mouse click."""
        try:
            self.mouse.click(Button.left, 1)
            return True
        except Exception as e:
            print(f"Error clicking mouse: {e}")
            return False

    def _scroll_up(self):
        """Simulate scrolling up."""
        try:
            self.mouse.scroll(0, 2)
            return True
        except Exception as e:
            print(f"Error scrolling up: {e}")
            return False

    def _scroll_down(self):
        """Simulate scrolling down."""
        try:
            self.mouse.scroll(0, -2)
            return True
        except Exception as e:
            print(f"Error scrolling down: {e}")
            return False

    def _open_google_search(self):
        """Open Google search in the default web browser."""
        try:
            webbrowser.open('https://www.google.com')
            return True
        except Exception as e:
            print(f"Error opening Google: {e}")
            return False

    def _attack(self):
        """Simulate an attack action (typically presses spacebar)."""
        try:
            self.keyboard.press(Key.space)
            self.keyboard.release(Key.space)
            return True
        except Exception as e:
            print(f"Error executing attack: {e}")
            return False

    def _back(self):
        """Simulate moving backward using the 'S' key."""
        try:
            # Use string 's' instead of Key.s for alphabetic keys
            self.keyboard.press('s')
            sleep(3)
            self.keyboard.release('s')
            return True
        except Exception as e:
            print(f"Error moving backward: {e}")
            return False

    def _forward(self):
        """Simulate moving forward using the 'W' key."""
        try:
            self.keyboard.press('w')
            sleep(3)
            self.keyboard.release('w')
            return True
        except Exception as e:
            print(f"Error moving forward: {e}")
            return False

    def _left(self):
        """Simulate moving left using the 'A' key."""
        try:
            self.keyboard.press('a')
            self.keyboard.release('a')
            return True
        except Exception as e:
            print(f"Error moving left: {e}")
            return False

    def _right(self):
        """Simulate moving right using the 'D' key."""
        try:
            self.keyboard.press('d')
            self.keyboard.release('d')
            return True
        except Exception as e:
            print(f"Error moving right: {e}")
            return False