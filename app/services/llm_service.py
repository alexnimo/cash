from typing import List, Dict, Any, Optional
import json
from app.core.config import get_settings
from app.services.model_manager import ModelManager, encode_image_to_base64
from app.utils.langtrace_utils import get_langtrace, trace_llm_call
import logging

# Initialize LangTrace at module level
_langtrace = get_langtrace()

logger = logging.getLogger(__name__)

settings = get_settings()

class LLMService:
    def __init__(self):
        """Initialize LLM service with configuration."""
        self.settings = get_settings()
        self._model_manager = None
        self._langtrace = _langtrace
        
    @property
    def model_manager(self):
        if self._model_manager is None:
            self._model_manager = ModelManager(settings)
        return self._model_manager
        
    @trace_llm_call("analyze_frames")
    async def analyze_frames(self, frames: List[str]) -> List[Dict[str, Any]]:
        """Analyze video frames using configured model"""
        logger.info(f"Starting frame analysis for {len(frames)} frames")
        client, model_config = await self.model_manager.get_video_analysis_model()
        results = []
        
        for frame in frames:
            logger.debug(f"Processing frame: {frame}")
            # Encode image to base64
            base64_image = encode_image_to_base64(frame)
            
            # Create the message with the image
            prompt = model_config.frame_analysis_prompt
            logger.debug("Frame analysis prompt created")
            
            try:
                if self._langtrace:
                    with self._langtrace.trace() as trace:
                        trace.add_metadata({
                            "function": "analyze_frames",
                            "frame_path": frame,
                            "prompt": prompt
                        })
                        logger.info(f"Starting frame analysis with trace_id: {trace.trace_id}")
                        response = await client.generate_content(
                            [prompt, base64_image],
                            trace_id=trace.trace_id
                        )
                else:
                    logger.debug("LangTrace not available, proceeding without tracing")
                    response = await client.generate_content([prompt, base64_image])

                if response.text:
                    result = {
                        "frame": frame,
                        "analysis": response.text,
                        "trace_id": trace.trace_id if self._langtrace else None
                    }
                    logger.info(f"Frame analysis successful: {frame}")
                    results.append(result)
                
            except Exception as e:
                logger.error(f"Error analyzing frame {frame}: {str(e)}", exc_info=True)
                results.append({
                    "frame": frame,
                    "error": str(e)
                })
                
        logger.info(f"Frame analysis completed. Processed {len(results)} frames")
        return results
    
    @trace_llm_call("transcribe_audio")
    async def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """Transcribe audio using configured model"""
        logger.info(f"Starting audio transcription for: {audio_path}")
        client, model_config = await self.model_manager.get_transcription_model()
        
        try:
            with open(audio_path, "rb") as audio_file:
                logger.debug("Audio file opened successfully")
                if self._langtrace:
                    with self._langtrace.trace() as trace:
                        logger.info(f"Starting transcription with trace_id: {trace.trace_id}")
                        response = await client.generate_content(
                            audio_file,
                            generation_config=model_config,
                            trace_id=trace.trace_id
                        )
                else:
                    logger.debug("LangTrace not available, proceeding without tracing")
                    response = await client.generate_content(
                        audio_file,
                        generation_config=model_config
                    )
                logger.info("Audio transcription completed successfully")
                return response.text
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}", exc_info=True)
            return None

    @trace_llm_call("generate_summary")
    async def generate_summary(self, transcript: str, frame_descriptions: List[Dict[str, Any]]) -> Optional[str]:
        """Generate a comprehensive summary using both transcript and frame analysis"""
        logger.info("Starting summary generation")
        client, model_config = await self.model_manager.get_video_analysis_model()
        
        try:
            # Combine transcript and frame descriptions
            context = "Video Content Analysis:\n\n"
            context += f"Transcript: {transcript}\n\nVisual Description:"
            for desc in frame_descriptions:
                context += f"\n- {desc.get('analysis', '')}"
            
            messages = [
                {
                    "role": "user",
                    "content": f"Based on the following video content, provide a comprehensive summary:\n\n{context}"
                }
            ]
            
            if self._langtrace:
                with self._langtrace.trace() as trace:
                    logger.info(f"Starting summary generation with trace_id: {trace.trace_id}")
                    response = await client.generate_content(
                        messages,
                        generation_config=model_config,
                        trace_id=trace.trace_id
                    )
            else:
                logger.debug("LangTrace not available, proceeding without tracing")
                response = await client.generate_content(
                    messages,
                    generation_config=model_config
                )
            
            logger.info("Summary generation completed successfully")
            return response.text
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}", exc_info=True)
            return None

    @trace_llm_call("generate_embeddings")
    async def generate_embeddings(self, text: str) -> Optional[List[float]]:
        """Generate embeddings using configured model"""
        logger.info("Starting embeddings generation")
        client, model_config = await self.model_manager.get_embedding_model()
        
        try:
            if self._langtrace:
                with self._langtrace.trace() as trace:
                    logger.info(f"Starting embeddings generation with trace_id: {trace.trace_id}")
                    response = await client.generate_content(
                        text,
                        generation_config=model_config,
                        trace_id=trace.trace_id
                    )
            else:
                logger.debug("LangTrace not available, proceeding without tracing")
                response = await client.generate_content(
                    text,
                    generation_config=model_config
                )
            
            logger.info("Embeddings generation completed successfully")
            return response.embedding
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
            return None

    @trace_llm_call("get_response")
    async def get_response(self, prompt: str, model_type: str = "chat") -> str:
        """Get response from LLM model."""
        logger.info(f"Starting response generation for prompt: {prompt}")
        try:
            model = await self.model_manager.get_model(model_type)
            
            # Use LangTrace if enabled
            if self._langtrace:
                with self._langtrace.trace() as trace:
                    logger.info(f"Starting response generation with trace_id: {trace.trace_id}")
                    response = await model.generate_content(
                        prompt,
                        trace_id=trace.trace_id
                    )
                    logger.info("Response generation completed successfully")
                    return response.text
            else:
                logger.debug("LangTrace not available, proceeding without tracing")
                response = await model.generate_content(prompt)
                logger.info("Response generation completed successfully")
                return response.text
                
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}", exc_info=True)
            raise Exception(f"Failed to get response: {str(e)}")

    def cleanup(self):
        """Clean up resources"""
        self.model_manager.cleanup()
