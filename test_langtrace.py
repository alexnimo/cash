import asyncio
import logging
import os
from dotenv import load_dotenv
from app.utils.langtrace_utils import init_langtrace, trace_gemini_call
from app.core.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@trace_gemini_call("test_gemini_call")
async def test_function():
    logger.info("Executing test function")
    return "Test response"

async def main():
    # Load environment variables
    load_dotenv()
    logger.info("Environment variables loaded")
    
    # Initialize LangTrace
    logger.info("Initializing LangTrace...")
    success = init_langtrace()
    logger.info(f"LangTrace initialization result: {success}")
    
    if not success:
        logger.error("LangTrace initialization failed")
        return
    
    # Test tracing
    logger.info("Testing LangTrace...")
    try:
        result = await test_function()
        logger.info(f"Test function result: {result}")
    except Exception as e:
        logger.error(f"Error in test function: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
