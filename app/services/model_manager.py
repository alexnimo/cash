from typing import Optional, Dict, Any, Tuple, List
import base64
from pathlib import Path
import os
import datetime
import google.generativeai as genai
from app.core.settings import get_settings
from app.services.model_config import configure_models
from app.utils.rate_limiter import RateLimiter
from app.utils.langtrace_utils import (
    trace_gemini_call, 
    get_langtrace, 
    init_langtrace, 
    setup_gemini
)
import logging
import time
import asyncio
from datetime import timedelta

settings = get_settings()
logger = logging.getLogger(__name__)

# Initialize LangTrace at module level
init_success = init_langtrace()
if init_success:
    logger.info("LangTrace initialized successfully at module level")
    # Set up Gemini integration
    if setup_gemini():
        logger.info("Gemini integration set up successfully")
    else:
        logger.warning("Failed to set up Gemini integration")
else:
    logger.warning("Failed to initialize LangTrace at module level")

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_audio_to_base64(audio_path: str) -> str:
    """Encode an audio file to base64."""
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')

class ModelManager:
    def __init__(self, settings):
        self.settings = settings
        self.models = {}
        self.model_configs = {
            settings.model.video_analysis.name: settings.model.video_analysis,
            settings.model.transcription.name: settings.model.transcription,
            settings.model.embedding.name: settings.model.embedding
        }
        
        # Initialize quota status
        self._quota_status = {
            settings.model.transcription.name: {
                'status': 'available',
                'error': None,
                'type': 'gemini-flash',
                'purpose': 'transcription',
                'last_error_time': None
            },
            settings.model.embedding.name: {
                'status': 'available',
                'error': None,
                'type': 'gemini-flash',
                'purpose': 'embedding',
                'last_error_time': None
            }
        }
        
        # Initialize rate limiter with configured RPM
        rpm = settings.api.gemini_rpm_override or settings.api.gemini_rpm
        self.rate_limiter = RateLimiter(
            max_requests=rpm,
            period=timedelta(minutes=1),
            callback=self._on_rate_limit
        )
        
        logger.info(f"Initialized ModelManager with Gemini RPM: {rpm} (interval: {60.0/rpm:.2f}s)")
        
    def _on_rate_limit(self):
        """Callback when rate limit is hit"""
        logger.warning("Rate limit reached, requests will be delayed")
        
    async def _wait_for_rate_limit(self, model_name: str) -> bool:
        """Wait for rate limit"""
        if not self.rate_limiter.allow_request():
            logger.debug(f"Rate limiting {model_name}: waiting for next available slot")
            await self.rate_limiter.wait()
        return True
        
    async def _initialize_model(self, model_name: str) -> Any:
        """Lazy initialization of models only when needed"""
        try:
            if model_name not in self.model_configs:
                raise ValueError(f"Unknown model: {model_name}")
                
            config = self.model_configs[model_name]
            
            # Configure the Gemini API
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            
            # Initialize the model with the specified configuration
            generation_config = {
                "temperature": config.temperature,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": config.max_tokens,
            }
            
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
            ]
            
            model = genai.GenerativeModel(
                model_name=config.name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            self.models[model_name] = model
            logger.info(f"Initialized model {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to initialize {model_name}: {str(e)}", exc_info=True)
            raise

    @trace_gemini_call("analyze_video_content")
    async def get_video_analysis_model(self):
        """Get or initialize video analysis model"""
        model_name = self.settings.model.video_analysis.name
        
        try:
            # Wait for rate limit before configuring
            if not await self._wait_for_rate_limit(model_name):
                raise ValueError(f"Rate limit exceeded for {model_name}")
            
            # Configure Gemini API
            genai.configure(api_key=self.settings.api.gemini_api_key)
            
            logger.info(f"Initializing video analysis model: {model_name}")
            model = genai.GenerativeModel(model_name)
            
            # Test the model with a simple prompt
            @trace_gemini_call("test_video_analysis_model")
            async def test_model():
                try:
                    response = model.generate_content("Test connection. Reply with 'OK'.")
                    return response
                except Exception as e:
                    logger.error(f"Test model failed: {str(e)}")
                    raise
            test_response = await test_model()
            if not test_response or not test_response.text or "OK" not in test_response.text.upper():
                raise ValueError(f"Model test failed. Response: {test_response}")
            
            self.models[model_name] = model
            logger.info(f"Successfully initialized {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to initialize {model_name}: {str(e)}", exc_info=True)
            raise

    @trace_gemini_call("generate_vision_content")
    async def get_vision_model(self):
        """Get or initialize vision model"""
        model_name = self.settings.model.video_analysis.name  # Use the model from config
        
        try:
            # Wait for rate limit before configuring
            if not await self._wait_for_rate_limit(model_name):
                raise ValueError(f"Rate limit exceeded for {model_name}")
            
            # Configure Gemini API
            genai.configure(api_key=self.settings.api.gemini_api_key)
            
            logger.info(f"Initializing vision model: {model_name}")
            model = genai.GenerativeModel(model_name)
            
            # Test the model with a simple text prompt (can't test vision without an image)
            @trace_gemini_call("test_vision_model")
            async def test_model():
                try:
                    response = model.generate_content("Test connection. Reply with 'OK'.")
                    return response
                except Exception as e:
                    logger.error(f"Test model failed: {str(e)}")
                    raise
            test_response = await test_model()
            if not test_response or not test_response.text or "OK" not in test_response.text.upper():
                raise ValueError(f"Model test failed. Response: {test_response}")
            
            self.models[model_name] = model
            logger.info(f"Successfully initialized {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to initialize {model_name}: {str(e)}", exc_info=True)
            raise

    @trace_gemini_call("generate_transcript")
    async def get_transcription_model(self) -> Tuple[Any, Dict[str, Any]]:
        """Get the model for audio transcription"""
        if not self.settings.model.transcription.enabled:
            raise ValueError("Transcription is not enabled")
            
        # Initialize model if not already initialized
        model = await self._initialize_model(self.settings.model.transcription.name)
        
        model_config = {
            "generation_config": {
                "temperature": 0.1,  # Low temperature for more accurate transcription
                "candidate_count": 1,
                "max_output_tokens": 1024,  # Increased for longer transcriptions
                "top_p": 0.8,
                "top_k": 40
            },
            "safety_settings": [
                {
                    "category": "HARM_CATEGORY_DANGEROUS",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                }
            ]
        }
        
        return model, model_config
    
    @trace_gemini_call("generate_embedding")
    async def get_embedding_model(self) -> Tuple[Any, Dict[str, Any]]:
        """Get the model for generating embeddings"""
        self._initialize_model(self.settings.model.embedding.name)
        model = self.models[self.settings.model.embedding.name]
        model_config = {
            "model": self.settings.model.embedding.name,
            "dimension": self.settings.model.embedding.dimension,
            "prompt": """
            Convert the following text into a numerical embedding vector.
            Output ONLY the space-separated floating point numbers representing the embedding.
            Do not include any descriptions, speaker labels, or additional formatting.
            The embedding should capture the semantic meaning of the text for later similarity matching.
            """
        }
        return model, model_config
    
    def _prepare_audio_input(self, audio_path: str) -> Tuple[Any, str]:
        """Prepare audio input for the model"""
        # For files under 4MB, use base64 encoding
        file_size = os.path.getsize(audio_path)
        if file_size < 4 * 1024 * 1024:  # 4MB
            with open(audio_path, "rb") as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
            return {"mime_type": "audio/mp3", "data": audio_base64}, "base64"
        else:
            # For larger files, use the File API
            return genai.upload_file(audio_path), "file_api"

    @trace_gemini_call("transcribe_audio")
    async def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using configured transcription model"""
        try:
            logger.debug(f"Starting transcription for audio file: {audio_path}")
            
            # Get model and config
            model, model_config = await self.get_transcription_model()
            logger.debug(f"Using model config: {model_config}")
            
            if not model:
                raise TranscriptionError("Transcription model not available")
            
            # Check quota status
            if self._quota_status[self.settings.model.transcription.name]['status'] != 'available':
                raise QuotaExceededError("Transcription quota exceeded")
            
            # Wait for rate limit
            if not await self._wait_for_rate_limit(self.settings.model.transcription.name):
                raise TranscriptionError("Rate limit exceeded")
            
            try:
                # Prepare the audio file
                logger.debug(f"Preparing audio file: {audio_path}")
                audio_input, input_type = self._prepare_audio_input(audio_path)
                logger.debug(f"Audio input prepared with type: {input_type}")
                
                # Generate transcript using the prepared input
                logger.debug("Generating transcript...")
                response = model.generate_content(
                    [audio_input, "Please transcribe this audio accurately, maintaining proper punctuation and speaker labels if multiple speakers are present."],
                    generation_config=model_config["generation_config"],
                    safety_settings=model_config["safety_settings"]
                )
                logger.debug(f"Raw response: {response}")
                
                if not response or not response.text:
                    raise TranscriptionError("Empty response from transcription model")
                
                transcript = response.text.strip()
                logger.debug(f"Generated transcript: {transcript[:100]}...")  # Log first 100 chars
                return transcript
                
            except Exception as e:
                logger.error(f"Error during transcription: {str(e)}", exc_info=True)
                raise TranscriptionError(f"Transcription failed: {str(e)}")
            
        except QuotaExceededError as e:
            logger.error(f"Quota exceeded for transcription: {str(e)}")
            self._update_quota_status(self.settings.model.transcription.name, error=e)
            raise
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {str(e)}", exc_info=True)
            self._update_quota_status(self.settings.model.transcription.name, error=e)
            raise TranscriptionError(f"Transcription failed: {str(e)}")

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using configured embedding model"""
        try:
            model, config = await self.get_embedding_model()
            
            @trace_gemini_call("generate_embedding")
            async def embed():
                return model.generate_content(
                    config["prompt"] + "\nText: " + text
                )
            
            response = await embed()
            
            if not response or not response.text:
                raise ValueError("No embedding generated")
            
            # Parse the embedding from the response
            embedding_str = response.text.strip()
            embedding = [float(x) for x in embedding_str.split()]
            
            if len(embedding) != config["dimension"]:
                raise ValueError(f"Generated embedding dimension {len(embedding)} does not match expected {config['dimension']}")
                
            return embedding
            
        except Exception as e:
            self._update_quota_status(self.settings.model.embedding.name, error=e)
            raise

    def _get_model_purpose(self, model_name):
        """Get the purpose of a model based on its configuration"""
        if model_name == self.settings.model.video_analysis.name:
            return "Video Analysis"
        elif model_name == self.settings.model.transcription.name:
            return "Transcription"
        elif model_name == self.settings.model.embedding.name:
            return "Embedding"
        return "Unknown"

    def _load_audio_file(self, audio_path: str) -> bytes:
        """Load an audio file"""
        with open(audio_path, "rb") as audio_file:
            return audio_file.read()

    def encode_audio_to_base64(self, audio_data: bytes) -> str:
        """Encode an audio file to base64"""
        return base64.b64encode(audio_data).decode('utf-8')

    def _update_quota_status(self, model_name, error=None):
        """Update quota status for a model based on API response"""
        if error:
            error_msg = str(error).lower()
            if any(err in error_msg for err in ["429", "quota", "exhausted", "rate limit"]):
                self._quota_status[model_name].update({
                    'status': 'exhausted',
                    'error': 'API quota exhausted',
                    'last_error_time': datetime.datetime.now()
                })
            else:
                self._quota_status[model_name].update({
                    'status': 'error',
                    'error': str(error),
                    'last_error_time': datetime.datetime.now()
                })
        else:
            # If no error, check if we should reset exhausted status
            if self._quota_status[model_name]['status'] == 'exhausted':
                last_error = self._quota_status[model_name]['last_error_time']
                if last_error and (datetime.datetime.now() - last_error).total_seconds() > 60:
                    # Reset after 1 minute
                    self._quota_status[model_name].update({
                        'status': 'available',
                        'error': None,
                        'last_error_time': None
                    })

    async def check_quota(self):
        """Check API quota status for gemini-flash models only"""
        try:
            # Update status for flash models that might have recovered
            flash_models = [name for name, info in self._quota_status.items() 
                          if info['type'] == 'gemini-flash']
            
            for model_name in flash_models:
                self._update_quota_status(model_name)
            
            return {
                'status': 'success',
                'quotas': {name: {k: v for k, v in info.items() if k != 'last_error_time'}
                          for name, info in self._quota_status.items()
                          if name in flash_models},
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking quota: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.datetime.now().isoformat()
            }

class QuotaExceededError(Exception):
    """Raised when API quota is exhausted"""
    pass

class TranscriptionError(Exception):
    """Raised when transcription fails"""
    pass
