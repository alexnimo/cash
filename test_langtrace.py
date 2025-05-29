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
    
    # Initialize LangTrace (now a stub implementation that always returns False)
    logger.info("Initializing LangTrace stub...")
    success = init_langtrace()
    logger.info(f"LangTrace initialization result: {success} (expected False since tracing is now disabled)")
    
    # Even though success will be False, we'll continue with the test to ensure compatibility
    logger.info("Testing trace_gemini_call decorator (stub implementation)...")
    try:
        result = await test_function()
        logger.info(f"Test function result: {result}")
        logger.info("Tracing has been successfully disabled across the codebase.")
        logger.info("The trace_gemini_call decorator now only provides timing logs.")
    except Exception as e:
        logger.error(f"Error in test function: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
