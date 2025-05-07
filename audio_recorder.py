import sounddevice as sd
import numpy as np
import wave
from datetime import datetime
from pathlib import Path
import scipy.signal as signal
import threading
import queue
import time

class AudioRecorder:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.recording = False
        self.frames = []
        self.recordings_dir = Path("recordings")
        self.recordings_dir.mkdir(exist_ok=True)
        # Improved buffer configuration for better listening
        self.buffer_duration = 3.0  # Increased buffer size for better capture
        self.extended_mode = False  # Flag for extended recordings (dictation)
        # Add voice activity detection parameters
        self.vad_enabled = True
        self.silence_threshold = 0.01
        self.vad_frame_duration = 0.03  # 30ms frames for VAD
        self.min_speech_duration = 0.3  # Minimum duration to consider valid speech
        # Real-time processing
        self.processing_queue = queue.Queue()
        self.continuous_listen = False
        self.energy_history = []
        self.energy_window = 20  # Number of frames for adaptive threshold

    def start_recording_async(self, extended_mode=False):
        """Start recording audio asynchronously
        
        Args:
            extended_mode: If True, optimizes for longer speech capture (dictation)
        """
        try:
            self.frames = []
            self.recording = True
            self.extended_mode = extended_mode
            self.energy_history = []
            
            # Adaptive silence threshold based on mode
            if extended_mode:
                self.silence_threshold = 0.005  # More sensitive for dictation
            else:
                self.silence_threshold = 0.008   # Improved sensitivity for commands
            
            def callback(indata, frames, time, status):
                if status:
                    print(f"Recording error: {status}")
                if self.recording:
                    # Make a copy of the data to avoid reference issues
                    frame_copy = indata.copy()
                    self.frames.append(frame_copy)
                    
                    # Calculate energy for adaptive threshold
                    frame_energy = np.mean(np.abs(frame_copy))
                    self.energy_history.append(frame_energy)
                    if len(self.energy_history) > self.energy_window:
                        self.energy_history.pop(0)
                        # Adapt threshold based on ambient noise level
                        if len(self.energy_history) > 5:  # Ensure we have enough samples
                            ambient_energy = np.percentile(self.energy_history, 10)
                            # Dynamic adjustment of silence threshold
                            self.silence_threshold = max(0.005, ambient_energy * 1.5)

                    # Submit frame for real-time processing if continuous listening enabled
                    if self.continuous_listen and not self.extended_mode:
                        self.processing_queue.put(frame_copy)

            # Configure settings based on recording mode
            blocksize = int(self.sample_rate * (0.5 if extended_mode else 0.05))  # Smaller blocks for faster response
            
            # Start recording with appropriate settings
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                callback=callback,
                blocksize=blocksize,
                latency='low'
            )
            self.stream.start()
            return True
        except Exception as e:
            print(f"Error starting recording: {e}")
            return False

    def start_continuous_listening(self):
        """Enable continuous listening mode with real-time processing"""
        self.continuous_listen = True
        threading.Thread(target=self._continuous_listen_processor, daemon=True).start()
        return True
        
    def stop_continuous_listening(self):
        """Disable continuous listening mode"""
        self.continuous_listen = False
        return True
        
    def _continuous_listen_processor(self):
        """Process audio frames in real-time for continuous listening"""
        buffer = []
        speech_detected = False
        silence_frames = 0
        
        while self.continuous_listen:
            try:
                # Get frame with timeout to allow checking if we should still be running
                frame = self.processing_queue.get(timeout=0.5)
                
                # Calculate energy
                frame_energy = np.mean(np.abs(frame))
                
                # Voice activity detection
                if frame_energy > self.silence_threshold:
                    if not speech_detected:
                        # Speech start detected
                        speech_detected = True
                        # Clear previous silence frames from buffer
                        if len(buffer) > 10:  # Keep some context
                            buffer = buffer[-10:]
                    
                    # Reset silence counter when we detect speech
                    silence_frames = 0
                    
                elif speech_detected:
                    # Count silence frames after speech
                    silence_frames += 1
                
                # Add frame to buffer
                buffer.append(frame)
                
                # If we detected speech followed by silence, process the segment
                if speech_detected and silence_frames >= 10:  # ~300ms of silence
                    speech_detected = False
                    if len(buffer) > 0:
                        # Process speech segment
                        audio_data = np.concatenate(buffer, axis=0)
                        threading.Thread(
                            target=self._process_speech_segment, 
                            args=(audio_data.copy(),), 
                            daemon=True
                        ).start()
                        # Reset buffer but keep some context
                        buffer = buffer[-5:]
                
                # Limit buffer size for memory efficiency
                if len(buffer) > 100:  # ~3 seconds at 30ms frames
                    buffer = buffer[-50:]
                    
            except queue.Empty:
                pass  # No new frames, continue loop
                
            except Exception as e:
                print(f"Error in continuous listening processor: {e}")
                time.sleep(0.1)  # Prevent tight loop on error
    
    def _process_speech_segment(self, audio_data):
        """Process a detected speech segment in the background"""
        try:
            # Apply preprocessing
            audio_data = self._preprocess_audio(audio_data, False)
            
            # Save as temporary file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.recordings_dir / f"temp_segment_{timestamp}.wav"
            
            # Only save segments with sufficient energy
            if np.max(np.abs(audio_data)) > 0.01 and len(audio_data) > self.sample_rate * 0.3:
                with wave.open(str(filename), 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.sample_rate)
                    # Convert float32 to int16
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    wav_file.writeframes(audio_int16.tobytes())
                
                # TODO: Here you could add a callback to submit this to your command processing system
                print(f"Speech segment detected! Saved to: {filename}")
        except Exception as e:
            print(f"Error processing speech segment: {e}")

    def stop_recording_and_save(self):
        """Stop recording and save the audio file with preprocessing"""
        try:
            if not self.recording:
                return None

            self.recording = False
            self.stream.stop()
            self.stream.close()

            if not self.frames or len(self.frames) == 0:
                print("No audio frames captured")
                return None

            # Combine all frames
            audio_data = np.concatenate(self.frames, axis=0)
            
            # Apply preprocessing - with mode-specific parameters
            audio_data = self._preprocess_audio(audio_data, self.extended_mode)

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

            print(f"Saved recording: {filename} (duration: {len(audio_data)/self.sample_rate:.2f}s)")
            return str(filename)
        except Exception as e:
            print(f"Error saving recording: {e}")
            return None

    def _preprocess_audio(self, audio_data, extended_mode=False):
        """Apply audio preprocessing to improve quality
        
        Args:
            audio_data: Raw audio data
            extended_mode: If True, uses settings optimized for dictation
        """
        try:
            # Skip processing if audio is too short
            if len(audio_data) < 0.1 * self.sample_rate:  # Less than 100ms
                return audio_data
                
            # Remove DC offset
            audio_data = audio_data - np.mean(audio_data)
            
            # Noise gate - completely silence very quiet parts
            noise_gate_threshold = 0.003
            noise_mask = np.abs(audio_data) < noise_gate_threshold
            audio_data[noise_mask] = 0.0
            
            # Normalize volume
            max_amplitude = np.max(np.abs(audio_data))
            if max_amplitude > 0.01:  # Only normalize if there's actual sound
                audio_data = audio_data / (max_amplitude + 1e-6)
                
            # Apply compression to boost voice
            audio_data = self._apply_compression(audio_data)
            
            # Apply high-pass filter to reduce low frequency noise
            nyquist = self.sample_rate / 2
            cutoff = 70 if extended_mode else 80  # Slightly lower cutoff for dictation
            b, a = signal.butter(4, cutoff / nyquist, btype='highpass')
            audio_data = signal.filtfilt(b, a, audio_data.flatten())
            
            # Apply low-pass filter to reduce high frequency noise
            cutoff_high = 7500 / nyquist
            b, a = signal.butter(6, cutoff_high, btype='lowpass')
            audio_data = signal.filtfilt(b, a, audio_data)
            
            # Different silence processing for extended mode vs command mode
            if not extended_mode:
                # Trim silence from beginning and end - only for command mode
                threshold = self.silence_threshold
                mask = np.abs(audio_data) > threshold
                if np.any(mask):
                    start = np.argwhere(mask)[0][0]
                    end = np.argwhere(mask)[-1][0] + 1
                    audio_data = audio_data[start:end]
                
                # Add small padding
                padding = np.zeros(int(0.1 * self.sample_rate))  # 100ms padding
                audio_data = np.concatenate([padding, audio_data, padding])
            else:
                # For dictation, just ensure we have enough content
                if len(audio_data) < self.sample_rate * 0.5:
                    # If too short, add padding
                    padding = np.zeros(int(0.5 * self.sample_rate) - len(audio_data))
                    audio_data = np.concatenate([audio_data, padding])
                    
            # Final normalization
            max_amplitude = np.max(np.abs(audio_data))
            if max_amplitude > 0.01:
                audio_data = audio_data / (max_amplitude + 1e-6) * 0.95  # Leave a bit of headroom
            
            return audio_data
            
        except Exception as e:
            print(f"Error in audio preprocessing: {e}")
            return audio_data  # Return original if preprocessing fails

    def _apply_compression(self, audio_data, threshold=0.1, ratio=3.0):
        """Apply dynamic range compression to boost quiet parts of speech
        
        Args:
            audio_data: Input audio data
            threshold: Compression threshold (0.0-1.0)
            ratio: Compression ratio
            
        Returns:
            Compressed audio data
        """
        try:
            # Simple compression algorithm
            result = np.zeros_like(audio_data)
            for i in range(len(audio_data)):
                amplitude = abs(audio_data[i])
                if amplitude > threshold:
                    # Apply compression above threshold
                    gain = threshold + (amplitude - threshold) / ratio
                    # Maintain sign
                    result[i] = np.sign(audio_data[i]) * gain
                else:
                    # Below threshold, keep original
                    result[i] = audio_data[i]
            
            return result
        except Exception as e:
            print(f"Error in audio compression: {e}")
            return audio_data
            
    def get_recordings_list(self):
        """Get list of all recordings"""
        try:
            return sorted(str(p) for p in self.recordings_dir.glob("*.wav"))
        except Exception as e:
            print(f"Error listing recordings: {e}")
            return []