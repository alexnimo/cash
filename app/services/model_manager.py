from typing import Optional, Dict, Any, Tuple, List, Union
import base64
from pathlib import Path
import os
import datetime
import google.generativeai as genai
from app.core.unified_config import get_config
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
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI

config = get_config()
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
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.model_configs = {}
        
        # Extract model configurations based on config type
        self._init_model_configs()
        
        # Initialize quota status
        self._quota_status = {}
        
        # Set up quota status for models
        self._init_quota_status()
        
        # Get RPM configuration
        rpm = self._get_rpm_config()
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            max_requests=rpm,
            period=timedelta(minutes=1),
            callback=self._on_rate_limit
        )
        
        logger.info(f"Initialized ModelManager with Gemini RPM: {rpm} (interval: {60.0/rpm:.2f}s)")
    
    def _init_model_configs(self):
        """Initialize model configurations directly from config"""
        # Store direct references to model configurations to avoid lookup by name
        self.video_analysis_config = None
        self.frame_analysis_config = None
        self.transcription_config = None
        
        # Get direct access to model configurations from config.yaml
        logger.info("Initializing model configurations from config")
        
        # Get video analysis model config
        if hasattr(self.config, 'get'):
            # Config is a ConfigManager instance
            self.video_analysis_config = self._get_model_section('video_analysis')
            self.frame_analysis_config = self._get_model_section('frame_analysis')
            self.transcription_config = self._get_model_section('transcription')
        elif isinstance(self.config, dict) and 'model' in self.config:
            # Config is a dictionary
            if 'video_analysis' in self.config['model']:
                self.video_analysis_config = self.config['model']['video_analysis']
            if 'frame_analysis' in self.config['model']:
                self.frame_analysis_config = self.config['model']['frame_analysis']
            if 'transcription' in self.config['model']:
                self.transcription_config = self.config['model']['transcription']
        elif hasattr(self.config, 'model'):
            # Config is an object with attributes
            if hasattr(self.config.model, 'video_analysis'):
                self.video_analysis_config = self.config.model.video_analysis
            if hasattr(self.config.model, 'frame_analysis'):
                self.frame_analysis_config = self.config.model.frame_analysis
            if hasattr(self.config.model, 'transcription'):
                self.transcription_config = self.config.model.transcription
                
        # Log the loaded configs
        if self.video_analysis_config:
            logger.info(f"Loaded video_analysis config: {self._get_config_summary(self.video_analysis_config)}")
            # Add to model_configs for backward compatibility
            if 'name' in self.video_analysis_config:
                self.model_configs[self.video_analysis_config['name']] = self.video_analysis_config
        
        if self.frame_analysis_config:
            logger.info(f"Loaded frame_analysis config: {self._get_config_summary(self.frame_analysis_config)}")
            # Add to model_configs for backward compatibility
            if 'name' in self.frame_analysis_config:
                self.model_configs[self.frame_analysis_config['name']] = self.frame_analysis_config
        
        if self.transcription_config:
            logger.info(f"Loaded transcription config: {self._get_config_summary(self.transcription_config)}")
            # Add to model_configs for backward compatibility
            if 'name' in self.transcription_config:
                self.model_configs[self.transcription_config['name']] = self.transcription_config
        else:
            logger.error("Failed to load transcription config from config file")
    
    def _get_model_section(self, section_name):
        """Get a model section from config using ConfigManager"""
        if not hasattr(self.config, 'get'):
            return None
            
        # Extract all relevant config as a dictionary
        config_dict = {}
        
        # Get base properties
        name = self.config.get('model', section_name, 'name')
        enabled = self.config.get('model', section_name, 'enabled', default=True)  
        model_type = self.config.get('model', section_name, 'type')
        temperature = self.config.get('model', section_name, 'temperature', default=0.7)
        thinking_budget = self.config.get('model', section_name, 'thinking_budget', default=0)
        
        if name:
            config_dict['name'] = name
            config_dict['enabled'] = enabled
            config_dict['type'] = model_type
            config_dict['temperature'] = temperature
            config_dict['thinking_budget'] = thinking_budget
            return config_dict
        return None
        
    def _get_config_summary(self, config):
        """Get a summary of a config object for logging"""
        if isinstance(config, dict):
            return config
        elif hasattr(config, '__dict__'):
            return {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
        else:
            return str(config)
                    
        logger.info(f"Initialized model_configs with {len(self.model_configs)} models: {list(self.model_configs.keys())}")
    
    def _init_quota_status(self):
        """Initialize quota status for models"""
        # For each model in model_configs, set up quota status
        for model_name, config in self.model_configs.items():
            purpose = 'general'
            if hasattr(config, 'purpose'):
                purpose = config.purpose
            elif isinstance(config, dict) and 'purpose' in config:
                purpose = config['purpose']
            elif 'transcription' in str(model_name).lower():
                purpose = 'transcription'
            elif 'frame' in str(model_name).lower():
                purpose = 'frame_analysis'
                
            self._quota_status[model_name] = {
                'status': 'available',
                'error': None,
                'type': 'gemini-flash',
                'purpose': purpose,
                'last_error_time': None
            }
    
    def _get_rpm_config(self):
        """Get RPM configuration from config"""
        rpm = 10  # Default RPM
        
        # Try attribute access
        if hasattr(self.config, 'api'):
            rpm = getattr(self.config.api, 'gemini_rpm_override', None) or getattr(self.config.api, 'gemini_rpm', rpm)
        # Try dictionary access
        elif isinstance(self.config, dict) and 'api' in self.config:
            rpm = self.config['api'].get('gemini_rpm_override') or self.config['api'].get('gemini_rpm', rpm)
        # Try ConfigManager get
        elif hasattr(self.config, 'get'):
            rpm = self.config.get('api', 'gemini_rpm_override', default=None) or self.config.get('api', 'gemini_rpm', default=rpm)
            
        return rpm

        
    def _on_rate_limit(self):
        """Callback when rate limit is hit"""
        logger.warning("Rate limit reached, requests will be delayed")
        
    async def _wait_for_rate_limit(self, model_name: str) -> bool:
        """Wait for rate limit and handle quota exceeded errors"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if not self.rate_limiter.allow_request():
                    logger.debug(f"Rate limiting {model_name}: waiting for next available slot")
                    await self.rate_limiter.wait()
                
                # Check quota status
                quota_status = self._quota_status.get(model_name, {})
                if quota_status.get('status') == 'exceeded':
                    last_error_time = quota_status.get('last_error_time')
                    if last_error_time:
                        time_since_error = datetime.datetime.now() - last_error_time
                        if time_since_error.total_seconds() < 60:  # Wait 60 seconds after quota exceeded
                            wait_time = 60 - time_since_error.total_seconds()
                            logger.warning(f"Quota exceeded for {model_name}, waiting {wait_time:.1f} seconds")
                            await asyncio.sleep(wait_time)
                
                return True
                
            except Exception as e:
                if "429" in str(e):
                    retry_count += 1
                    wait_time = 60  # Wait 60 seconds on 429 error
                    logger.warning(f"Rate limit (429) hit for {model_name}, retry {retry_count}/{max_retries} after {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                raise
        
        raise QuotaExceededError(f"Max retries ({max_retries}) exceeded for {model_name}")
        
    async def _initialize_model(self, model_name: str) -> Any:
        """Lazily initialize models only when they are needed using model_configs"""
        try:
            # Check if model already exists in cache
            if model_name in self.models:
                logger.debug(f"Using cached model: {model_name}")
                return self.models[model_name]
            
            # If model not in model_configs, add it with default settings
            if model_name not in self.model_configs:
                logger.warning(f"Model {model_name} not found in model_configs, using default settings")
                self.model_configs[model_name] = {
                    'name': model_name,
                    'temperature': 0.7 if 'vision' in model_name or 'pro' in model_name else 0.2
                }
            
            config = self.model_configs[model_name]
            
            # Configure the Gemini API
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            
            # Initialize the model with the specified configuration
            # Handle both dictionary and attribute style config objects
            temperature = None
            thinking_budget = None
            
            # Try attribute access first
            if hasattr(config, 'temperature'):
                temperature = config.temperature
            elif isinstance(config, dict) and 'temperature' in config:
                temperature = config['temperature']
            else:
                temperature = 0.7
                
            if hasattr(config, 'thinking_budget'):
                thinking_budget = config.thinking_budget
            elif isinstance(config, dict) and 'thinking_budget' in config:
                thinking_budget = config['thinking_budget']
            else:
                thinking_budget = 0
                
            # Log model initialization
            logger.info(f"Initializing model {model_name} with temperature={temperature}")
            
            # Create the model
            model = genai.GenerativeModel(model_name=model_name)
            self.models[model_name] = model
            
            return model
        except Exception as e:
            logger.error(f"Failed to initialize model {model_name}: {e}")
            raise

    async def _initialize_model_directly(self, model_name: str, model_config: dict) -> Any:
        """Initialize a model directly with the provided configuration"""
        try:
            # Check if model already exists in cache
            if model_name in self.models:
                logger.debug(f"Using cached model: {model_name}")
                return self.models[model_name]
            
            # Configure the Gemini API
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            
            # Initialize the model with the specified configuration
            temperature = model_config.get('temperature', 0.7)
            thinking_budget = model_config.get('thinking_budget', 0)
            
            # Log model initialization with detailed config
            logger.info(f"Initializing model {model_name} with config: {model_config}")
            
            # Create the model
            model = genai.GenerativeModel(model_name=model_name)
            
            # Store thinking_budget in model metadata for later use
            model._thinking_budget = thinking_budget
            
            # Cache the model for future use
            self.models[model_name] = model
            
            return model
        except Exception as e:
            logger.error(f"Failed to initialize model {model_name}: {e}")
            raise

    @trace_gemini_call("analyze_video_content")
    async def get_video_analysis_model(self):
        """Get or initialize video analysis model"""
        model_name = self.config.model.video_analysis.name if hasattr(self.config, 'model') and hasattr(self.config.model, 'video_analysis') else "gemini-pro"
        
        try:
            # Wait for rate limit before configuring
            if not await self._wait_for_rate_limit(model_name):
                raise ValueError(f"Rate limit exceeded for {model_name}")
            
            # Configure Gemini API
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if hasattr(self.config, 'api') and hasattr(self.config.api, 'gemini_api_key'):
                gemini_api_key = self.config.api.gemini_api_key
            genai.configure(api_key=gemini_api_key)
            
            logger.info(f"Initializing video analysis model: {model_name}")
            model = await self._initialize_model(model_name)
            
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
        model_name = self.config.model.video_analysis.name if hasattr(self.config, 'model') and hasattr(self.config.model, 'video_analysis') else "gemini-pro-vision"
        
        try:
            # Wait for rate limit before configuring
            if not await self._wait_for_rate_limit(model_name):
                raise ValueError(f"Rate limit exceeded for {model_name}")
            
            # Configure Gemini API
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if hasattr(self.config, 'api') and hasattr(self.config.api, 'gemini_api_key'):
                gemini_api_key = self.config.api.gemini_api_key
            genai.configure(api_key=gemini_api_key)
            
            logger.info(f"Initializing vision model: {model_name}")
            model = await self._initialize_model(model_name)
            
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
        # Check if transcription config is available
        if not self.transcription_config or not self.transcription_config.get('name'):
            raise ValueError("Transcription configuration is missing or invalid. Check config.yaml.")
        
        # Get model name directly from stored config
        model_name = self.transcription_config['name']
        logger.debug(f"Using transcription model: {model_name}")
        
        # Initialize the model
        model = await self._initialize_model_directly(model_name, self.transcription_config)
        
        model_config = {
            "generation_config": {
                "temperature": 0.1,  # Low temperature for more accurate transcription
                "candidate_count": 1,
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
        
    def get_transcription_model_instance(self) -> Any:
        """Get just the model instance for transcription (without config) for use with chunking"""
        # Check if transcription config is available
        if not self.transcription_config or not self.transcription_config.get('name'):
            raise ValueError("Transcription configuration is missing or invalid. Check config.yaml.")
        
        # Get model name directly from stored config
        model_name = self.transcription_config['name']
        temperature = self.transcription_config.get('temperature', 0.1)
        
        logger.debug(f"Creating transcription model instance: {model_name} (temp: {temperature})")
        
        # Create the model directly from our stored config
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": temperature,
                "candidate_count": 1,
                "top_p": 0.8,
                "top_k": 40
            },
            safety_settings=[
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
        )
        
        return model
    
    @trace_gemini_call("frame_analysis")
    async def get_frame_analysis_model(self) -> Any:
        """Get or initialize frame analysis model"""
        # Use the config attribute instead of settings
        model_name = self.config.model.frame_analysis.name if hasattr(self.config, 'model') and hasattr(self.config.model, 'frame_analysis') else 'gemini-2.5-flash-preview-04-17'
        
        try:
            # Wait for rate limit before configuring
            if not await self._wait_for_rate_limit(model_name):
                raise ValueError(f"Rate limit exceeded for {model_name}")
            
            # Configure Gemini API
            api_key = self.config.api.gemini_api_key if hasattr(self.config, 'api') and hasattr(self.config.api, 'gemini_api_key') else os.getenv('GEMINI_API_KEY')
            genai.configure(api_key=api_key)
            
            logger.info(f"Initializing frame analysis model: {model_name}")
            await self._initialize_model(model_name)
            model = self.models.get(model_name)
            
            if not model:
                raise ValueError(f"Failed to initialize frame analysis model: {model_name}")
                
            logger.info(f"Successfully initialized {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to initialize {model_name}: {str(e)}", exc_info=True)
            raise
    
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

            # Upload file using Gemini's file API
            try:
                audio_file = genai.upload_file(audio_path, mime_type="audio/wav")
                logger.info(f"Successfully uploaded audio file: {audio_path}")
            except Exception as e:
                logger.error(f"Failed to upload audio file: {str(e)}")
                raise TranscriptionError(f"Failed to upload audio file: {str(e)}")

            # Handle both dictionary-style access and ConfigManager object access
            model_name = None
            temperature = 0.1  # Default temperature
            
            # Try to get model name using ConfigManager's get method if available
            if hasattr(self.config, 'get'):
                model_name = self.config.get('model', 'transcription', 'name')
                if model_name:
                    temperature = self.config.get('model', 'transcription', 'temperature', default=0.1)
            # Try dictionary-style access
            elif isinstance(self.config, dict) and 'model' in self.config and 'transcription' in self.config['model']:
                config_transcription = self.config['model']['transcription']
                model_name = config_transcription.get('name')
                if model_name:
                    temperature = config_transcription.get('temperature', 0.1)
            # Try attribute-style access as a last resort
            elif hasattr(self.config, 'model') and hasattr(self.config.model, 'transcription'):
                model_name = getattr(self.config.model.transcription, 'name', None)
                if model_name:
                    temperature = getattr(self.config.model.transcription, 'temperature', 0.1)
            
            if not model_name:
                raise ValueError("Transcription model name not found in config")
                
            # Get stored model config if available
            stored_model_config = self.model_configs.get(model_name)
            
            if not stored_model_config:
                # Use defaults if no config found in model_configs
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config={
                        "temperature": 0.1,
                        "candidate_count": 1,
                        "top_p": 0.8,
                        "top_k": 40
                    }
                )
            else:
                model = genai.GenerativeModel(
                    model_name=stored_model_config.name,
                generation_config={
                    "temperature": stored_model_config.temperature or 0.1,
                    "candidate_count": 1,
                    "top_p": 0.8,
                    "top_k": 40
                },
                safety_settings=model_config["safety_settings"]
            )

            # Create transcription prompt
            prompt = """
            Your task is to provide a complete and accurate transcription of the entire audio file.
            Requirements:
            1. Transcribe ALL spoken content from start to finish
            2. Do not skip or summarize any parts
            3. Format as plain text without timestamps or speaker labels
            4. Maintain word-for-word accuracy
            5. Include every single word that is spoken
            
            Important: The transcription must be complete and cover the entire duration of the audio.
            """

            try:
                # Generate transcription with streaming to handle longer content
                response = model.generate_content([prompt, audio_file], stream=True, request_options={"timeout": 1000})
                
                # Collect all response chunks
                transcript_parts = []
                for chunk in response:
                    if chunk.text:
                        transcript_parts.append(chunk.text.strip())
                
                # Combine all parts
                transcript = " ".join(transcript_parts)
                
                logger.info(f"Successfully transcribed audio file: {audio_path}")
                return transcript

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg:
                    logger.error("Rate limit exceeded during transcription")
                    raise QuotaExceededError("Rate limit exceeded")
                else:
                    logger.error(f"Transcription failed: {error_msg}")
                    raise TranscriptionError(f"Transcription failed: {error_msg}")

        except Exception as e:
            if isinstance(e, (QuotaExceededError, TranscriptionError)):
                raise
            logger.error(f"Unexpected error during transcription: {str(e)}")
            raise TranscriptionError(f"Unexpected error during transcription: {str(e)}")

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using video analysis model as fallback"""
        try:
            # Use video analysis model instead since embedding model was removed
            model = await self.get_video_analysis_model()
            
            dimension = 1024  # Default dimension
            prompt = """
            Convert the following text into a numerical embedding vector.
            Output ONLY the space-separated floating point numbers representing the embedding.
            Do not include any descriptions, speaker labels, or additional formatting.
            The embedding should capture the semantic meaning of the text for later similarity matching.
            """
            
            @trace_gemini_call("generate_embedding")
            async def embed():
                return model.generate_content(
                    prompt + "\nText: " + text
                )
            
            response = await embed()
            
            if not response or not response.text:
                raise ValueError("No embedding generated")
            
            # Parse the embedding from the response
            embedding_str = response.text.strip()
            embedding = [float(x) for x in embedding_str.split()]
            
            if len(embedding) != dimension:
                raise ValueError(f"Generated embedding dimension {len(embedding)} does not match expected {dimension}")
                
            return embedding
            
        except Exception as e:
            self._update_quota_status(self.settings.model.video_analysis.name, error=e)
            raise

    def _get_model_purpose(self, model_name):
        """Get the purpose of a model based on its configuration"""
        if model_name == self.settings.model.video_analysis.name:
            return "Video Analysis"
        elif model_name == self.settings.model.frame_analysis.name:
            return "Frame Analysis"
        elif model_name == self.settings.model.transcription.name:
            return "Transcription"
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
        if model_name not in self._quota_status:
            return
            
        if error is None:
            # Reset status if call was successful
            self._quota_status[model_name].update({
                'status': 'available',
                'error': None,
                'last_error_time': None
            })
        else:
            error_str = str(error).lower()
            if "429" in error_str or "quota" in error_str or "resource exhausted" in error_str:
                self._quota_status[model_name].update({
                    'status': 'exceeded',
                    'error': str(error),
                    'last_error_time': datetime.datetime.now()
                })
                logger.warning(f"Quota exceeded for {model_name}: {error}")
            else:
                self._quota_status[model_name].update({
                    'status': 'error',
                    'error': str(error),
                    'last_error_time': datetime.datetime.now()
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

    async def upload_file(self, file_path: Union[str, Path]) -> Any:
        """Upload a file to Gemini using the file upload API"""
        try:
            # Convert string to Path if needed
            if isinstance(file_path, str):
                file_path = Path(file_path)
                
            logger.info(f"Uploading file to Gemini: {file_path}")
            if not file_path.exists():
                raise ValueError(f"File not found: {file_path}")
            
            # Upload file using Gemini's file upload API
            uploaded_file = genai.upload_file(str(file_path))
            logger.info(f"Successfully uploaded file: {file_path}")
            return uploaded_file
            
        except Exception as e:
            logger.error(f"Failed to upload file to Gemini: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def create_llm(config: dict):
        """Create LLM instance based on config"""
        provider = config.get('provider', 'gemini')
        model = config.get('model', 'models/gemini-2.0-flash-exp')
        
        if provider.lower() == 'gemini':
            return Gemini(
                model_name=model,
                api_key=os.getenv("GEMINI_API_KEY"),
                temperature=config.get('temperature', 0.3),
                max_tokens=config.get('max_tokens', 32000)
            )
        elif provider.lower() == 'openai':
            return OpenAI(
                model=model,
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=config.get('temperature', 0.3),
                max_tokens=config.get('max_tokens', 32000)
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

class QuotaExceededError(Exception):
    """Raised when API quota is exhausted"""
    pass

class TranscriptionError(Exception):
    """Raised when transcription fails"""
    pass
