"""
ElevenLabs Text-to-Speech Service

This module provides text-to-speech functionality using ElevenLabs API.
"""

import os
import requests
import uuid
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ElevenLabsTTS:
    """ElevenLabs Text-to-Speech service"""
    
    def __init__(self, api_key: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM"):
        """
        Initialize ElevenLabs TTS service
        
        Args:
            api_key: ElevenLabs API key
            voice_id: Voice ID to use (default: Rachel)
        """
        self.api_key = api_key
        self.voice_id = voice_id
        self.base_url = "https://api.elevenlabs.io/v1"
        
        if not self.api_key:
            raise ValueError("ElevenLabs API key is required")
    
    def get_available_voices(self) -> Dict[str, Any]:
        """
        Get list of available voices from ElevenLabs
        
        Returns:
            Dictionary containing voice information
        """
        url = f"{self.base_url}/voices"
        headers = {
            "Accept": "application/json",
            "xi-api-key": self.api_key
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get voices: {e}")
            raise Exception(f"Failed to get available voices: {e}")
    
    def generate_speech(
        self, 
        text: str, 
        voice_id: Optional[str] = None,
        model_id: str = "eleven_monolingual_v1",
        voice_settings: Optional[Dict[str, float]] = None
    ) -> bytes:
        """
        Generate speech from text using ElevenLabs API
        
        Args:
            text: Text to convert to speech
            voice_id: Voice ID to use (optional, uses default if not provided)
            model_id: Model ID to use for generation
            voice_settings: Voice settings (stability, similarity_boost, style, use_speaker_boost)
            
        Returns:
            Audio data as bytes
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Use provided voice_id or default
        voice_id = voice_id or self.voice_id
        
        # Default voice settings
        if voice_settings is None:
            voice_settings = {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True
            }
        
        url = f"{self.base_url}/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        data = {
            "text": text,
            "model_id": model_id,
            "voice_settings": voice_settings
        }
        
        try:
            logger.info(f"Generating speech for text length: {len(text)} characters")
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code != 200:
                error_msg = f"ElevenLabs API error: {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f" - {error_detail}"
                except:
                    error_msg += f" - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            audio_data = response.content
            logger.info(f"Successfully generated audio: {len(audio_data)} bytes")
            return audio_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise Exception(f"Failed to generate speech: {e}")
    
    def save_audio_to_file(self, audio_data: bytes, output_path: str) -> str:
        """
        Save audio data to file
        
        Args:
            audio_data: Audio data as bytes
            output_path: Path to save the audio file
            
        Returns:
            Path to the saved file
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, "wb") as f:
                f.write(audio_data)
            
            logger.info(f"Audio saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save audio file: {e}")
            raise Exception(f"Failed to save audio file: {e}")
    
    def generate_and_save_speech(
        self, 
        text: str, 
        output_dir: str = "./temp_audio",
        voice_id: Optional[str] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Generate speech and save to file
        
        Args:
            text: Text to convert to speech
            output_dir: Directory to save the audio file
            voice_id: Voice ID to use (optional)
            filename: Custom filename (optional, generates UUID if not provided)
            
        Returns:
            Path to the saved audio file
        """
        # Generate filename if not provided
        if filename is None:
            filename = f"speech_{uuid.uuid4()}.mp3"
        elif not filename.endswith('.mp3'):
            filename += '.mp3'
        
        output_path = os.path.join(output_dir, filename)
        
        # Generate speech
        audio_data = self.generate_speech(text, voice_id)
        
        # Save to file
        return self.save_audio_to_file(audio_data, output_path)

# Convenience function for quick usage
def create_elevenlabs_tts(api_key: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM") -> ElevenLabsTTS:
    """
    Create ElevenLabs TTS instance
    
    Args:
        api_key: ElevenLabs API key
        voice_id: Voice ID to use
        
    Returns:
        ElevenLabsTTS instance
    """
    return ElevenLabsTTS(api_key, voice_id)
