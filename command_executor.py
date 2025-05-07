import subprocess
import os
import time
import threading
from pynput.keyboard import Key, Controller
from pynput.mouse import Button, Controller as MouseController
import webbrowser
import requests
import json
from command_queue import CommandQueue

class CommandExecutor:
    def __init__(self):
        self.keyboard = Controller()
        self.mouse = MouseController()
        self.audio_recorder = None  # Will be set by the calling code
        
        # Set up command queue system
        self.command_queue = CommandQueue(default_delay=0.1)
        self.command_queue.set_executor(self._execute_command_internal)
        
        self.commands = {
            "open notepad": self._open_notepad,
            "open calculator": self._open_calculator,
            "close window": self._close_window,
            "click": self._click,
            "scroll up": self._scroll_up,
            "scroll down": self._scroll_down,
            "open google search": self._open_google_search,
            "clear queue": self._clear_command_queue,
            # New navigation commands
            "forward": self._move_forward,
            "back": self._move_back,
            "left": self._move_left,
            "right": self._move_right,
            "attack": self._attack
        }
        print(f"Available commands: {list(self.commands.keys())}")

    def set_audio_recorder(self, recorder):
        """Set the audio recorder to use"""
        self.audio_recorder = recorder
        
    def register_queue_callback(self, event_type, callback):
        """Register a callback for command queue events"""
        self.command_queue.register_callback(event_type, callback)

    def execute_command(self, command_text):
        """Execute the corresponding command based on the recognized text."""
        # Handle multiple commands
        if isinstance(command_text, list):
            print(f"Adding {len(command_text)} commands to queue")
            self.command_queue.add_command(command_text)
            return True
            
        command_text = command_text.lower().strip()
        print(f"Attempting to execute command: '{command_text}'")
        
        # Add command to queue
        if command_text in self.commands:
            print(f"Adding command to queue: '{command_text}'")
            self.command_queue.add_command(command_text)
            return True
        else:
            print(f"No matching command found for: '{command_text}'")
            return False
            
    def _execute_command_internal(self, command_text):
        """Internal execution function called by the command queue"""
        if command_text in self.commands:
            print(f"Executing command: '{command_text}'")
            success = self.commands[command_text]()
            print(f"Command execution {'succeeded' if success else 'failed'}")
            return success
        return False
        
    def _clear_command_queue(self):
        """Clear all pending commands in queue"""
        try:
            self.command_queue.clear()
            return True
        except Exception as e:
            print(f"Error clearing command queue: {e}")
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

    def _move_forward(self):
        """Simulate pressing the W key for forward movement."""
        try:
            self.keyboard.press('w')
            time.sleep(0.3)  # Hold for a short duration
            self.keyboard.release('w')
            return True
        except Exception as e:
            print(f"Error moving forward: {e}")
            return False
            
    def _move_back(self):
        """Simulate pressing the S key for backward movement."""
        try:
            self.keyboard.press('s')
            time.sleep(0.3)
            self.keyboard.release('s')
            return True
        except Exception as e:
            print(f"Error moving back: {e}")
            return False
            
    def _move_left(self):
        """Simulate pressing the A key for left movement."""
        try:
            self.keyboard.press('a')
            time.sleep(0.3)
            self.keyboard.release('a')
            return True
        except Exception as e:
            print(f"Error moving left: {e}")
            return False
            
    def _move_right(self):
        """Simulate pressing the D key for right movement."""
        try:
            self.keyboard.press('d')
            time.sleep(0.3)
            self.keyboard.release('d')
            return True
        except Exception as e:
            print(f"Error moving right: {e}")
            return False
            
    def _attack(self):
        """Simulate a left mouse click for attacking."""
        try:
            self.mouse.click(Button.left, 1)
            return True
        except Exception as e:
            print(f"Error performing attack: {e}")
            return False