# Voice Command System

A Python-based voice command system that allows users to control their computer using voice commands. The system uses OpenAI's Whisper model for speech recognition and provides a user-friendly GUI for recording, training, and executing voice commands.

## Features

- **Speech Recognition**: Uses OpenAI's Whisper model for accurate speech-to-text conversion
- **Command Recording**: Record individual or bulk voice commands with audio preprocessing
- **Command Training**: Train the system to recognize your voice commands
- **Command Execution**: Execute system commands like opening applications or closing windows
- **User-friendly GUI**: Three separate interfaces for recording, training, and testing
- **Audio Preprocessing**: Includes noise reduction, silence trimming, and volume normalization
- **GPU Acceleration**: CUDA support for faster speech recognition

## Requirements

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (optional but recommended)
- Windows OS (for the current command set)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/daanishmittal24/voice_to_command.git
cd voice-to-command
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python main.py
python startup.py
```

2. The main window will open with three modes:

### Recording Mode
- Record individual voice commands
- Bulk record multiple samples of the same command
- View and manage recorded commands
- Audio preprocessing for better quality

### Training Mode
- View available training data
- Train the model on recorded commands
- Monitor training progress
- Save trained models

### Testing/Deployment Mode
- Test voice commands in real-time
- View command recognition results
- Execute recognized commands
- Monitor command execution status

## Available Commands

Currently supported commands:
- `open notepad`: Opens Windows Notepad
- `open calculator`: Opens Windows Calculator
- `close window`: Closes the active window (Alt+F4)

## Project Structure

- `startup.py`: Main entry point and GUI initialization
- `model_trainer.py`: Whisper model management and training
- `audio_recorder.py`: Audio recording and preprocessing
- `command_executor.py`: Command execution logic
- `command_organizer.py`: Command data management
- `gui_modes.py`: GUI implementation for different modes
- `recordings/`: Directory for storing recorded audio
- `command_data/`: Directory for organized command recordings
- `trained_models/`: Directory for model checkpoints and commands

## Technical Details

### Audio Processing
- Sample Rate: 16000 Hz
- Channels: Mono
- Format: WAV (16-bit)
- Preprocessing:
  - DC offset removal
  - Volume normalization
  - High-pass filtering (80 Hz cutoff)
  - Silence trimming
  - Padding (100ms)

### Speech Recognition
- Model: OpenAI Whisper (tiny)
- Features:
  - Command-specific prompt engineering
  - Multi-level command matching:
    1. Exact match
    2. Keyword match
    3. Fuzzy match
  - Temperature control for consistent results
  - Beam search optimization

### GUI
- Built with Tkinter
- Three separate modes:
  1. Recording interface
  2. Training interface
  3. Testing/Deployment interface
- Real-time feedback
- Progress monitoring
- Error handling

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

[Add your license here]

## Acknowledgments

- OpenAI Whisper for speech recognition
- PyTorch for deep learning suppozrt
- Libraries used: librosa, sounddevice, pynput
