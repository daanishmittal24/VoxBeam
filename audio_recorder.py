import sounddevice as sd
import numpy as np
import wave
from datetime import datetime
from pathlib import Path
import scipy.signal as signal

class AudioRecorder:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.recording = False
        self.frames = []
        self.recordings_dir = Path("recordings")
        self.recordings_dir.mkdir(exist_ok=True)

    def start_recording_async(self):
        """Start recording audio asynchronously"""
        try:
            self.frames = []
            self.recording = True
            
            def callback(indata, frames, time, status):
                if status:
                    print(f"Recording error: {status}")
                if self.recording:
                    self.frames.append(indata.copy())

            # Start recording with noise reduction settings
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                callback=callback,
                latency='low'
            )
            self.stream.start()
            return True
        except Exception as e:
            print(f"Error starting recording: {e}")
            return False

    def stop_recording_and_save(self):
        """Stop recording and save the audio file with preprocessing"""
        try:
            if not self.recording:
                return None

            self.recording = False
            self.stream.stop()
            self.stream.close()

            if not self.frames:
                return None

            # Combine all frames
            audio_data = np.concatenate(self.frames, axis=0)
            
            # Apply preprocessing
            audio_data = self._preprocess_audio(audio_data)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.recordings_dir / f"recording_{timestamp}.wav"

            # Save as WAV file
            with wave.open(str(filename), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                # Convert float32 to int16
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())

            return str(filename)
        except Exception as e:
            print(f"Error saving recording: {e}")
            return None

    def _preprocess_audio(self, audio_data):
        """Apply audio preprocessing to improve quality"""
        try:
            # Remove DC offset
            audio_data = audio_data - np.mean(audio_data)
            
            # Normalize volume
            audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-6)
            
            # Apply high-pass filter to reduce low frequency noise
            nyquist = self.sample_rate / 2
            cutoff = 80  # Hz
            b, a = signal.butter(4, cutoff / nyquist, btype='highpass')
            audio_data = signal.filtfilt(b, a, audio_data.flatten())
            
            # Trim silence from beginning and end
            threshold = 0.01
            mask = np.abs(audio_data) > threshold
            if np.any(mask):
                start = np.argwhere(mask)[0][0]
                end = np.argwhere(mask)[-1][0] + 1
                audio_data = audio_data[start:end]
            
            # Add small padding
            padding = np.zeros(int(0.1 * self.sample_rate))  # 100ms padding
            audio_data = np.concatenate([padding, audio_data, padding])
            
            return audio_data
            
        except Exception as e:
            print(f"Error in audio preprocessing: {e}")
            return audio_data  # Return original if preprocessing fails

    def get_recordings_list(self):
        """Get list of all recordings"""
        try:
            return sorted(str(p) for p in self.recordings_dir.glob("*.wav"))
        except Exception as e:
            print(f"Error listing recordings: {e}")
            return []