import os
import requests
import json
import time
from pathlib import Path
import base64
import threading

class OnlineTranscriptionService:
    """Handles online speech recognition using various free API services"""
    
    def __init__(self):
        # Configuration 
        self.enabled = True
        self.services = ["assemblyai", "deepgram", "azure"]  # Supported services
        self.default_service = "assemblyai"  # Default service to use
        self.api_keys = {
            "assemblyai": os.environ.get("ASSEMBLYAI_API_KEY", ""),
            "deepgram": os.environ.get("DEEPGRAM_API_KEY", ""),
            "azure": os.environ.get("AZURE_SPEECH_KEY", "")
        }
        self.azure_region = os.environ.get("AZURE_SPEECH_REGION", "eastus")
        
        # Rate limiting
        self.request_lock = threading.Lock()
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds
        
    def is_available(self, service=None):
        """Check if an online transcription service is available"""
        service = service or self.default_service
        return self.enabled and self.api_keys.get(service, "") != ""
        
    def transcribe_file(self, audio_file, service=None):
        """Transcribe an audio file using online service
        
        Args:
            audio_file: Path to audio file
            service: Service to use (or default if None)
            
        Returns:
            Text transcription or None if failed
        """
        service = service or self.default_service
        
        # Rate limiting
        with self.request_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
            self.last_request_time = time.time()
            
        if not self.is_available(service):
            print(f"Online transcription service '{service}' not available")
            return None
            
        try:
            # Select appropriate service
            if service == "assemblyai":
                return self._transcribe_with_assemblyai(audio_file)
            elif service == "deepgram":
                return self._transcribe_with_deepgram(audio_file)
            elif service == "azure":
                return self._transcribe_with_azure(audio_file)
            else:
                print(f"Unknown service: {service}")
                return None
        except Exception as e:
            print(f"Error in online transcription with {service}: {e}")
            return None
            
    def _transcribe_with_assemblyai(self, audio_file):
        """Transcribe using AssemblyAI"""
        try:
            api_key = self.api_keys["assemblyai"]
            if not api_key:
                return None
                
            # Upload file
            headers = {
                "authorization": api_key,
                "content-type": "application/json"
            }
            
            # First upload the file
            with open(audio_file, 'rb') as f:
                upload_response = requests.post(
                    "https://api.assemblyai.com/v2/upload",
                    headers=headers,
                    data=f
                )
            upload_url = upload_response.json()["upload_url"]
            
            # Then submit for transcription
            data = {
                "audio_url": upload_url,
                "language_code": "en",
            }
            transcription_response = requests.post(
                "https://api.assemblyai.com/v2/transcript",
                json=data,
                headers=headers
            )
            
            transcript_id = transcription_response.json()["id"]
            
            # Poll for completion
            polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
            status = "processing"
            
            while status != "completed":
                polling_response = requests.get(polling_endpoint, headers=headers)
                status = polling_response.json()["status"]
                
                if status == "error":
                    raise Exception("Transcription failed")
                    
                if status != "completed":
                    time.sleep(1)
                    
            result = polling_response.json()["text"]
            return result
                
        except Exception as e:
            print(f"AssemblyAI transcription error: {e}")
            return None
            
    def _transcribe_with_deepgram(self, audio_file):
        """Transcribe using Deepgram"""
        try:
            api_key = self.api_keys["deepgram"]
            if not api_key:
                return None
                
            headers = {
                "Authorization": f"Token {api_key}",
                "Content-Type": "audio/wav"
            }
            
            with open(audio_file, 'rb') as f:
                audio_data = f.read()
                
            url = "https://api.deepgram.com/v1/listen?model=nova&language=en&detect_language=false"
            response = requests.post(url, headers=headers, data=audio_data)
            
            if response.status_code == 200:
                result = response.json()
                return result["results"]["channels"][0]["alternatives"][0]["transcript"]
            else:
                print(f"Deepgram error: {response.status_code}, {response.text}")
                return None
                
        except Exception as e:
            print(f"Deepgram transcription error: {e}")
            return None
            
    def _transcribe_with_azure(self, audio_file):
        """Transcribe using Azure Speech Service"""
        try:
            api_key = self.api_keys["azure"]
            region = self.azure_region
            if not api_key or not region:
                return None
                
            # Full implementation would use Azure SDK
            # This is a simplified placeholder
            print("Azure Speech Service support is not fully implemented")
            return None
                
        except Exception as e:
            print(f"Azure transcription error: {e}")
            return None