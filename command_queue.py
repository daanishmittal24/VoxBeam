import threading
import time
import queue

class CommandQueue:
    """
    Manages a queue of commands to be executed with visual feedback
    and configurable delays between commands.
    """
    def __init__(self, default_delay=0.1):  # Reduced delay from 1.5s to 0.1s
        self.command_queue = queue.Queue()
        self.processing_thread = None
        self.is_processing = False
        self.default_delay = default_delay  # Minimal delay between commands
        self.current_command = None
        self.callbacks = {
            "on_command_start": [],
            "on_command_complete": [],
            "on_queue_empty": [],
            "on_queue_update": []
        }
        
    def add_command(self, command):
        """Add a command or list of commands to the queue"""
        if isinstance(command, list):
            # Add multiple commands
            for cmd in command:
                self.command_queue.put(cmd)
        else:
            # Add single command
            self.command_queue.put(command)
            
        # Notify listeners about queue update
        self._notify("on_queue_update", self.get_queue_status())
        
        # Start processing if not already running
        self._ensure_processing()
        
    def get_queue_status(self):
        """Get the current queue status as a dictionary"""
        try:
            queue_items = list(self.command_queue.queue)
        except:
            queue_items = []
        
        return {
            "current_command": self.current_command,
            "queue_size": self.command_queue.qsize(),
            "next_commands": queue_items[:5],  # Show up to 5 upcoming commands
            "is_processing": self.is_processing
        }
        
    def register_callback(self, event_type, callback_fn):
        """Register a callback for a specific event type"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback_fn)
        else:
            raise ValueError(f"Unknown event type: {event_type}")
            
    def _notify(self, event_type, data=None):
        """Notify all callbacks registered for an event type"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"Error in {event_type} callback: {e}")
                
    def _process_commands(self):
        """Process commands from the queue with minimal delay between them"""
        self.is_processing = True
        
        try:
            while not self.command_queue.empty():
                # Get next command
                command = self.command_queue.get()
                self.current_command = command
                
                # Notify about queue update
                self._notify("on_queue_update", self.get_queue_status())
                
                # Notify about command start
                self._notify("on_command_start", command)
                
                # Let executor know we're ready for next command
                executor_callback = self.callbacks.get("executor", None)
                if executor_callback:
                    executor_callback(command)
                
                # Minimal wait to prevent system overload and allow UI updates
                if self.default_delay > 0:
                    time.sleep(self.default_delay)
                
                # Notify about command completion
                self._notify("on_command_complete", command)
                
                # Mark task as done in the queue
                self.command_queue.task_done()
                
            # Queue is empty
            self.current_command = None
            self._notify("on_queue_empty")
            
        finally:
            self.is_processing = False
            
    def _ensure_processing(self):
        """Ensure the processing thread is running if needed"""
        if not self.is_processing and not self.command_queue.empty():
            self.processing_thread = threading.Thread(
                target=self._process_commands,
                daemon=True
            )
            self.processing_thread.start()
            
    def set_executor(self, executor_callback):
        """Set the function that will execute commands"""
        self.callbacks["executor"] = executor_callback
        
    def clear(self):
        """Clear all commands from the queue"""
        # Create a new queue to replace the existing one
        while not self.command_queue.empty():
            try:
                self.command_queue.get(block=False)
                self.command_queue.task_done()
            except queue.Empty:
                break
                
        self._notify("on_queue_update", self.get_queue_status())