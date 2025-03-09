from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
import json
import time
import asyncio
import uuid
import os
import copy
import traceback
import datetime
import numpy as np

from llama_index.core import VectorStoreIndex, QueryBundle, Document, Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata, BaseTool, FunctionTool
from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole, LLM
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.prompts import PromptTemplate
import pinecone
from pinecone import Pinecone, ServerlessSpec
import agentops
from agentops import track_agent

try:
    from langchain.agents import create_react_agent
    from langchain.agents.agent import AgentExecutor
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
except ImportError as e:
    logging.error(f"Error importing langchain: {str(e)}")

agentops_api_key = os.getenv("AGENTOPS_API_KEY")

from app.tools import notion_tool_v2
from app.tools.notion_tool_v2 import NotionTool
from app.config.agent_config import AGENT_CONFIG
from app.llm import LLMFactory

logger = logging.getLogger(__name__)
agentops.init(agentops_api_key)

def normalize_path(path: str) -> str:
    """
    Normalize path to work across different platforms.
    Will attempt to handle both Windows and Unix-style paths.
    """
    # Convert to Path object and back to string to normalize
    return str(Path(path))

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
        
    return dot_product / (norm_vec1 * norm_vec2)

def parse_date(date_str: str) -> datetime.datetime:
    """Parse date string into datetime object"""
    if not date_str:
        return datetime.datetime.now()
        
    try:
        import re
        date_match = re.search(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})', date_str)
        if date_match:
            date_part = date_match.group(1)
            sep = '-' if '-' in date_part else '/'
            parts = date_part.split(sep)
            return datetime.datetime(int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        pass
        
    # Default to current time if parsing fails
    return datetime.datetime.now()

class BaseFinanceAgent(ReActAgent):
    """Base class for finance-specific agents"""
    def __init__(self, tools: List[BaseTool], llm: Any, memory: ChatMemoryBuffer, config: Dict[str, Any]):
        chat_formatter = ReActChatFormatter(
            system_header=config.get('system_prompt', '')
        )

        super().__init__(
            tools=tools,
            llm=llm,
            memory=memory,
            max_iterations=config.get('max_iterations', 10),
            react_chat_formatter=chat_formatter,
            verbose=True
        )

class TechnicalAnalysisAgent(BaseFinanceAgent):
    """Agent for technical analysis tasks"""
    ANALYSIS_PROMPT = """Act as an Expert Technical Analysis Consolidator. Transform fragmented stock commentary into institutional-grade technical summaries following these precise instructions:

        **Core Mission**
        Create comprehensive technical profiles for tracked stocks by synthesizing all relevant data points across the provided analysis materials.

        **Critical Processing Rules**
        1. STRICT FILTERING:
           - Process ONLY tracked stocks that have actual data in the input report
           - EXCLUDE tracked stocks that have no meaningful data or mentions
           - EXCLUDE all non-tracked stocks entirely
        2. DATA PRESERVATION:
           - Maintain ALL visual references (frame_paths)
           - Preserve chronological sequence of events
           - Keep ALL relevant technical indicators and patterns

        **Technical Analysis Protocol**
        For each valid stock mention:
        1. Data Extraction:
           - Price Action: Historical movements, key reversals, trend structure
           - Volume Analysis: Trading activity, liquidity patterns, accumulation/distribution
           - Technical Indicators: Moving averages, RSI, MACD, etc.
           - Chart Patterns: Support/resistance, trend lines, formations
           
        2. Risk Assessment:
           - Volatility Profile: Historical and expected volatility levels
           - Risk Factors: Technical weaknesses, overhead resistance, dilution risks
           - Liquidity Considerations: Trading volume, market depth
           
        3. Opportunity Analysis:
           - Technical Setup Quality: Pattern completeness and reliability
           - Entry/Exit Levels: Key price points for trade management
           - Catalyst Timeline: Upcoming events that could impact technicals

        **Synthesis Requirements**
        Construct comprehensive technical profiles with:
        1. Price Architecture:
           - Key support/resistance levels
           - Trend structure and momentum
           - Price pattern formations
           
        2. Risk/Reward Matrix:
           - Potential reward targets
           - Clear risk levels
           - Position sizing considerations
           
        3. Action Framework:
           - Technical triggers for entry
           - Risk management levels
           - Target price objectives

        **Output Format**
        Deliver a strict JSON response:
        {
            "Date": "<current_date>",
            "Channel name": "<channel_name>",
            "sections": [
                {
                    "topic": "Technical Profile: <STOCK>",
                    "stocks": ["<STOCK>"],
                    "frame_paths": ["relevant_chart_paths"],
                    "source": "Composite Analysis",
                    "summary": "[Technical Context] + [Setup Quality] + [Risk/Reward Profile]",
                    "key_points": [
                        "Pattern: <identified_technical_pattern>",
                        "Setup: <current_technical_setup>",
                        "Risk: <key_risk_levels>",
                        "Target: <price_objectives>",
                        "Trigger: <entry_signals>"
                    ]
                }
            ]
        }

        IMPORTANT: Only include sections for tracked stocks that have actual technical analysis data in the input. Do not create empty or placeholder sections."""

    def __init__(self, tools: List[BaseTool], llm: LLM, memory: ChatMemoryBuffer, config: Dict[str, Any]):
        # Store LLM before initializing base agent
        self.llm_service = llm
        
        # Initialize base agent
        super().__init__(tools=tools, llm=llm, memory=memory, config=config)
        
        self.config = config
        self.debug_dir = Path(AGENT_CONFIG['debug_dir'])
        self.debug_dir.mkdir(exist_ok=True)
        
        # Initialize RAG agent
        self.rag_agent = RAGAgent(
            config=AGENT_CONFIG.get('rag', {}),
            llm_service=llm
        )
        
        # Initialize Notion tool for tracked stocks
        self.notion_tool = next(t for t in tools if isinstance(t, NotionTool))
        
        config['system_prompt'] = self.ANALYSIS_PROMPT

    async def execute(self, analysis_data: Union[Dict, str, Path]) -> Dict:
        """Execute technical analysis workflow"""
        try:
            # Save input data for debugging
            debug_file = self.debug_dir / f"ta_input_{int(time.time())}.json"
            with open(debug_file, "w") as f:
                json.dump(analysis_data, f, indent=2)
            logger.info(f"Saved technical analysis input to {debug_file}")

            # Get tracked stocks from Notion
            tracked_stocks = await self.notion_tool.get_all_tickers()
            
            # Fix formatting: Ensure tracked_stocks is a proper list of strings
            if isinstance(tracked_stocks, str):
                # If somehow returned as a string, convert to list
                tracked_stocks = [stock.strip() for stock in tracked_stocks.split(',') if stock.strip()]
            
            # Create a properly formatted string representation
            tracked_stocks_str = ", ".join([str(stock) for stock in tracked_stocks])
            logger.info(f"Retrieved tracked stocks: {tracked_stocks_str}")

            # Process input data
            if isinstance(analysis_data, Path):
                with open(analysis_data, 'r') as f:
                    market_content = f.read()
            elif isinstance(analysis_data, str):
                market_content = analysis_data
            else:
                market_content = json.dumps(analysis_data, indent=2)

            # Create analysis prompt
            prompt = f"""{self.ANALYSIS_PROMPT}

            Input Analysis Data:
            {market_content}

            Tracked Stocks: {tracked_stocks_str}
            """

            # Generate consolidated report using LLM
            if hasattr(self.llm_service, 'generate_text'):
                response = await asyncio.to_thread(
                    self.llm_service.generate_text,
                    prompt
                )
            else:
                response = await asyncio.to_thread(
                    self.llm_service.complete,
                    prompt
                )

            # Process response
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            try:
                # Try to find JSON in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start == -1 or json_end == 0:
                    logger.error("No JSON found in LLM response")
                    raise ValueError("LLM response does not contain valid JSON")
                    
                json_str = response_text[json_start:json_end]
                consolidated_report = json.loads(json_str)
                
                # Validate required fields
                if not isinstance(consolidated_report, dict):
                    raise ValueError("Consolidated report must be a dictionary")
                    
                required_fields = ["Date", "Channel name", "sections"]
                missing_fields = [field for field in required_fields if field not in consolidated_report]
                if missing_fields:
                    raise ValueError(f"Missing required fields in report: {missing_fields}")
                    
                if not isinstance(consolidated_report.get("sections"), list):
                    raise ValueError("Sections must be a list")
                    
                # Save consolidated report for debugging
                consolidated_debug_file = self.debug_dir / f"ta_consolidated_{int(time.time())}.json"
                with open(consolidated_debug_file, "w") as f:
                    json.dump(consolidated_report, f, indent=2)
                logger.info(f"Saved consolidated report to {consolidated_debug_file}")

                # Pass to RAG agent for processing
                rag_result = await self.rag_agent.execute(consolidated_report)
                
                return rag_result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
                raise ValueError(f"Invalid JSON in LLM response: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing LLM response: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Error in technical analysis execution: {str(e)}")
            raise

class MarketAnalysisAgent(BaseFinanceAgent):
    """Agent for market analysis tasks"""
    def __init__(self, tools: List[BaseTool], llm: LLM, memory: ChatMemoryBuffer, config: Dict[str, Any]):
        self.config = config
        self.debug_dir = Path(AGENT_CONFIG['debug_dir'])
        self.debug_dir.mkdir(exist_ok=True)
        
        # Initialize RAG agent
        self.rag_agent = RAGAgent(
            config=AGENT_CONFIG.get('rag', {}),
            llm_service=llm
        )
        
        # Initialize Notion tool for tracked stocks
        self.notion_tool = next(t for t in tools if isinstance(t, NotionTool))
        
        # Initialize base agent
        super().__init__(tools=tools, llm=llm, memory=memory, config=config)
        
        config['system_prompt'] = """Act as a Stock Analysis Consolidator. Transform fragmented stock commentary into institutional-grade technical summaries using this strict protocol:

        **Core Objective**  
        Create unified technical profiles for tracked stocks by synthesizing all mentions across source materials.

        **Input Processing Rules**  
        1. Filter mercilessly - EXCLUDE all non-tracked stocks  
        2. Preserve ALL visual references (frame_paths)  
        3. Maintain chronological event sequence
        4. Do not include sections for stocks not mentioned in the input report

        **Consolidation Protocol**  
        For each stock:  
        - Merge ALL entries into single profile  
        - Extract technical parameters from narratives:  
        - Price history markers (e.g., "$0.7→$10 post-announcement")  
        - Volatility catalysts (policy changes, earnings, M&A rumors)  
        - Liquidity signals (volume patterns, float analysis)  
        - Structural levels (IPO price, historical support/resistance analogs)  
        - Risk multipliers (dilution potential, short interest cues)  

        **Synthesis Requirements**  
        Construct 3-element technical profiles:  
        1. **Price Architecture** - Map historical extremes and reaction levels  
        2. **Event Horizon** - Identify upcoming volatility triggers  
        3. **Liquidity Matrix** - Assess trading viability and exit risks  

        **Output Format**  
        Deliver strict JSON with:  
        {
        "Date": "",
        "Channel name": "",
        "sections": [
            {
            "topic": "Technical Profile: {STOCK}",
            "stocks": ["{STOCK}"],
            "frame_paths": ["path1", ...],  
            "source": "Composite Analysis",
            "summary": "[Price Context] + [Volatility Profile] + [Key Risk/Return Ratio]",
            "key_points": [
                "Pattern: {HistoricalPriceBehavior}", 
                "Trigger: {Catalyst}",
                "Risk: {StructuralWeakness}",
                "Level: {CriticalPriceThreshold}"
            ]
            }
        ]
        }"""

    async def execute(self, market_data: Union[Dict, str, Path]) -> Dict:
        """Execute market analysis workflow"""
        try:
            # Save input data for debugging
            debug_file = self.debug_dir / f"market_input_{int(time.time())}.json"
            with open(debug_file, "w") as f:
                json.dump(market_data, f, indent=2)
            logger.info(f"Saved market input data to {debug_file}")

            # Get tracked stocks from Notion
            tracked_stocks = await self.notion_tool.get_all_tickers()
            
            # Fix formatting: Ensure tracked_stocks is a proper list of strings
            if isinstance(tracked_stocks, str):
                # If somehow returned as a string, convert to list
                tracked_stocks = [stock.strip() for stock in tracked_stocks.split(',') if stock.strip()]
            
            # Create a properly formatted string representation
            tracked_stocks_str = ", ".join([str(stock) for stock in tracked_stocks])
            logger.info(f"Retrieved tracked stocks: {tracked_stocks_str}")

            # Process input data
            if isinstance(market_data, Path):
                with open(market_data, 'r') as f:
                    market_content = f.read()
            elif isinstance(market_data, str):
                market_content = market_data
            else:
                market_content = json.dumps(market_data, indent=2)

            # Create distillation prompt
            prompt = f"""Act as a Stock Analysis Consolidator. Transform this market data into a structured technical summary.

            **Tracked Stocks to Include**
            {tracked_stocks_str}

            **Market Data Content**
            {market_content}

            Follow these strict rules:
            1. ONLY include analysis for the tracked stocks listed above
            2. Maintain ALL frame_paths references
            3. Follow the exact JSON output structure from my system prompt
            4. Ensure chronological sequence of events
            5. Extract all technical parameters as specified
            6. Construct complete technical profiles with all required elements

            Generate the consolidated report now."""

            # Generate consolidated report using LLM
            if hasattr(self.llm, 'generate_text'):
                response = await asyncio.to_thread(
                    self.llm.generate_text,
                    prompt
                )
            else:
                response = await asyncio.to_thread(
                    self.llm.complete,
                    prompt
                )

            # Process response
            if hasattr(response, 'text'):
                consolidated_report = json.loads(response.text)
            else:
                consolidated_report = json.loads(str(response))
            
            # Save consolidated report for debugging
            consolidated_debug_file = self.debug_dir / f"market_consolidated_{int(time.time())}.json"
            with open(consolidated_debug_file, "w") as f:
                json.dump(consolidated_report, f, indent=2)

            # Pass to RAG agent for processing
            rag_result = await self.rag_agent.execute(consolidated_report)
            
            return rag_result

        except Exception as e:
            logger.error(f"Error in market analysis execution: {str(e)}")
            raise

    async def _generate_consolidated_report(self, analysis_data: Dict, tracked_stocks: List[str]) -> Dict:
        """Generate consolidated report for tracked stocks only"""
        try:
            # Filter sections to include only tracked stocks
            filtered_sections = []
            for section in analysis_data.get('sections', []):
                section_stocks = section.get('stocks', [])
                tracked_section_stocks = [s for s in section_stocks if s in tracked_stocks]
                if tracked_section_stocks:
                    section['stocks'] = tracked_section_stocks
                    filtered_sections.append(section)

            return {
                "Date": analysis_data.get('Date', ''),
                "Channel name": analysis_data.get('Channel name', ''),
                "sections": filtered_sections
            }
        except Exception as e:
            logger.error(f"Error generating consolidated report: {str(e)}")
            raise

@track_agent(name="RAGagent")
class RAGAgent:
    """Agent for RAG operations"""
    
    def __init__(self, config: Dict[str, Any], llm_service: Any):
        self.config = config
        self.llm_service = llm_service
        self.debug_dir = Path(AGENT_CONFIG['debug_dir'])
        self.debug_dir.mkdir(exist_ok=True)
        self.similarity_threshold = config.get('similarity_threshold', 0.85)
        
        # Initialize vector store and embedding model
        self._init_vector_store()
        
        # Initialize Notion agent for updates
        self.notion_agent = NotionAgent(
            tools=[NotionTool()],
            llm=llm_service,
            memory=ChatMemoryBuffer.from_defaults(),
            config=AGENT_CONFIG.get('notion', {})
        )

        system_prompt_lines = [
            "You are an expert Stock Analysis Data Manager. Your task is to intelligently manage stock analysis data in a vector database by deciding whether to UPDATE, ADD NEW, or take NO ACTION on stock analysis sections.",
            "",
            "ANALYSIS CONTEXT:",
            "For each section, you will receive:",
            "1. NEW SECTION:",
            "   - Text content combining topic, stocks, summary, and key points",
            "   - Metadata including date, channel_name, stocks, and other attributes",
            "2. SIMILAR EXISTING SECTION (if found):",
            "   - Text content and metadata from vector database",
            "3. Similarity Score between the sections",
            "",
            "DECISION GUIDELINES:",
            "",
            "1. ADD NEW (Similarity Score < 0.85):",
            "   - When no similar section exists",
            "   - When similarity score indicates substantially different content",
            "   Justification required: Adding new analysis on stocks from channel_name dated date",
            "",
            "2. UPDATE (Similarity Score >= 0.85):",
            "   Requirements:",
            "   a) New section's date is more recent than existing section",
            "   b) Content provides meaningful updates:",
            "      - New key points",
            "      - Updated price targets",
            "      - New technical patterns",
            "      - Additional frame images",
            "   Justification required: Updating analysis on stocks with new data from channel_name dated date",
            "",
            "3. NO ACTION:",
            "   Conditions:",
            "   - Similar content exists with more recent date",
            "   - New content doesn't provide significant updates",
            "   - New content is older than existing analysis",
            "   Justification required: No update needed - specific reason",
            "",
            "EVALUATION CRITERIA:",
            "1. Date Comparison:",
            "   - Compare dates between new and existing sections",
            "   - Consider market relevance timeframes",
            "",
            "2. Content Value Assessment:",
            "   - Technical analysis updates",
            "   - Price level changes",
            "   - New market catalysts",
            "   - Risk factor updates",
            "   - Trading volume patterns",
            "",
            "3. Quality Factors:",
            "   - Comprehensiveness of analysis",
            "   - Specificity of price targets",
            "   - Supporting evidence (charts, data)",
            "   - Technical indicator analysis",
            "",
            "Your output should be a clear decision (ADD, UPDATE, or NO ACTION) with a concise justification referencing the specific criteria used."
        ]
        
        # Join the lines with newlines to create the complete system prompt
        system_prompt_main = "\n".join(system_prompt_lines)
        
        # Update the system prompt in the config with all parts
        self.config['system_prompt'] = system_prompt_main
        
        # Initialize vector store and embedding model
        self._init_vector_store()
        
        # Initialize Notion agent for updates
        self.notion_agent = NotionAgent(
            tools=[NotionTool()],
            llm=llm_service,
            memory=ChatMemoryBuffer.from_defaults(),
            config=AGENT_CONFIG.get('notion', {})
        )

        self.system_prompt = system_prompt_main

        self.system_prompt = """You are an expert Stock Analysis Data Manager. Your task is to intelligently manage stock analysis data in a vector database by deciding whether to UPDATE, ADD NEW, or take NO ACTION on stock analysis sections.

ANALYSIS CONTEXT:
For each section, you will receive:
1. NEW SECTION:
   - Text content combining topic, stocks, summary, and key points
   - Metadata including date, channel_name, stocks, and other attributes
2. SIMILAR EXISTING SECTION (if found):
   - Text content and metadata from vector database
3. Similarity Score between the sections

DECISION GUIDELINES:

1. ADD NEW (Similarity Score < 0.85):
   - When no similar section exists
   - When similarity score indicates substantially different content
   Justification required: Adding new analysis on stocks from channel_name dated date

2. UPDATE (Similarity Score >= 0.85):
   Requirements:
   a) New section's date is more recent than existing section
   b) Content provides meaningful updates:
      - New key points
      - Updated price targets
      - New technical patterns
      - Additional frame images
   Justification required: Updating analysis on stocks with new data from channel_name dated date

3. NO ACTION:
   Conditions:
   - Similar content exists with more recent date
   - New content doesn't provide significant updates
   - New content is older than existing analysis
   Justification required: No update needed - specific reason

EVALUATION CRITERIA:
1. Date Comparison:
   - Compare dates between new and existing sections
   - Consider market relevance timeframes
   
2. Content Value Assessment:
   - Technical analysis updates
   - Price level changes
   - New market catalysts
   - Risk factor updates
   - Trading volume patterns
   
3. Quality Factors:
   - Comprehensiveness of analysis
   - Specificity of price targets
   - Supporting evidence (charts, data)
   - Technical indicator analysis

Your output should be a clear decision (ADD, UPDATE, or NO ACTION) with a concise justification referencing the specific criteria used."""

    def _init_vector_store(self):
        from app.tools.pinecone_tool_v2 import PineconeAdvancedToolSpec
        from llama_index.vector_stores.pinecone import PineconeVectorStore
        self.pinecone_tool = PineconeAdvancedToolSpec()
        self.vector_store = PineconeVectorStore(pinecone_index=self.pinecone_tool.index, embedding_model=self.pinecone_tool.embed_model)

    def _prepare_section_text(self, section: Dict[str, Any]) -> str:
        """Prepare text for vectorization from a section."""
        components = [
            f"Topic: {section.get('topic', '')}",
            f"Stocks: {', '.join(section.get('stocks', []))}",
            f"Summary: {section.get('summary', '')}",
            f"Key Points: {'. '.join(section.get('key_points', []))}"
        ]
        return "\n".join(components)

    def _prepare_section_metadata(self, section: Dict[str, Any], report_data: Dict[str, Any], section_index: int) -> Dict[str, Any]:
        """Prepare metadata for a section."""
        return {
            "date": report_data.get('Date', ''),
            "channel_name": report_data.get('Channel name', ''),
            "topic": section.get('topic', ''),
            "stocks": section.get('stocks', []),
            "source": section.get('source', ''),
            "frame_paths": section.get('frame_paths', []),
            "section_index": section_index
        }

    def _parse_date(self, date_str: str) -> datetime.datetime:
        """Parse date string to datetime object."""
        try:
            return datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            try:
                return datetime.datetime.strptime(date_str, "%B %d %Y")
            except ValueError:
                logger.warning(f"Could not parse date: {date_str}")
                return datetime.datetime.min

    async def _process_section(self, section: Dict[str, Any], report_data: Dict[str, Any], section_index: int) -> Optional[Dict[str, Any]]:
        """Process a single section and determine if it needs to be updated."""
        try:
            # Prepare text and metadata
            section_text = self._prepare_section_text(section)
            metadata = self._prepare_section_metadata(section, report_data, section_index)
            
            # Get stocks from the section for filtering
            stocks = metadata["stocks"]
            if not stocks:
                logger.warning("No stocks found in section, skipping")
                return None
                
            # Generate embedding
            embedding_model = self.pinecone_tool.get_embedding_model()
            new_embedding = await embedding_model._aget_text_embedding(section_text)
            
            # First, pull all existing data related to these stocks from the vector store
            all_related_data = []
            for ticker in stocks:
                filter_dict = {
                    "stocks": {"$in": [ticker]}
                }
                
                stock_docs = await self.pinecone_tool.query_similar(
                    f"Stock information about {ticker}",
                    top_k=10,  # Get more results to ensure comprehensive comparison
                    filter=filter_dict
                )
                
                if stock_docs:
                    all_related_data.extend(stock_docs)
                    
            logger.info(f"Found {len(all_related_data)} existing records for stocks: {stocks}")
            
            # Query for similar sections with metadata filters
            filter_dict = {
                "$and": [
                    {"stocks": {"$in": metadata["stocks"]}}
                ]
            }
            
            similar_docs = await self.pinecone_tool.query_similar(
                section_text,
                top_k=1,
                filter=filter_dict
            )

            if similar_docs:
                existing_doc = similar_docs[0]
                existing_text = self._prepare_section_text(existing_doc)
                existing_embedding = await embedding_model._aget_text_embedding(existing_text)
                
                # Calculate similarity
                similarity = cosine_similarity(new_embedding, existing_embedding)
                
                # Compare dates if similarity is high
                if similarity >= self.similarity_threshold:
                    new_date = self._parse_date(metadata["date"])
                    existing_date = self._parse_date(existing_doc.get("date", ""))
                    
                    # Update if newer date OR if content is significantly different despite high similarity
                    if new_date > existing_date:
                        logger.info(f"Updating section for {metadata['stocks']} due to newer date (similarity: {similarity:.2f})")
                        return section
                    else:
                        # Check for meaningful content differences
                        content_changes = self._detect_content_changes(section, existing_doc)
                        if content_changes:
                            logger.info(f"Updating section for {metadata['stocks']} due to content changes: {content_changes}")
                            return section
                        else:
                            logger.info(f"Skipping update for {metadata['stocks']} - no significant changes (similarity: {similarity:.2f})")
                            return None
                else:
                    logger.info(f"Low similarity ({similarity:.2f}) for {metadata['stocks']}, treating as new content")
                    return section
            else:
                logger.info(f"No existing content found for {metadata['stocks']}, adding as new")
                return section

        except Exception as e:
            logger.error(f"Error processing section: {str(e)}")
            return None

    async def _process_section_v2(self, section: Dict[str, Any], report_data: Dict[str, Any], section_index: int) -> Optional[Dict[str, Any]]:
        """Process a single section with enhanced re-ranking and content merging logic."""
        try:
            # Prepare text and metadata
            section_text = self._prepare_section_text(section)
            metadata = self._prepare_section_metadata(section, report_data, section_index)
            
            # Get stocks from the section for filtering
            stocks = metadata.get("stocks", [])
            if not stocks:
                logger.warning("No stocks found in section, skipping")
                return None
                
            logger.info(f"Processing section with stocks: {stocks}")
            
            # Generate embedding for current section
            embedding_model = self.pinecone_tool.get_embedding_model()
            new_embedding = await embedding_model._aget_text_embedding(section_text)
            
            # Pull all existing data related to these stocks from the vector store
            all_related_data = []
            for ticker in stocks:
                filter_dict = {
                    "stocks": {"$in": [ticker]}
                }
                
                stock_docs = await self.pinecone_tool.query_similar(
                    f"Stock information about {ticker}",
                    top_k=10,  # Get more results to ensure comprehensive comparison
                    filter=filter_dict
                )
                
                if stock_docs:
                    all_related_data.extend(stock_docs)
                    
            logger.info(f"Found {len(all_related_data)} existing records for stocks: {stocks}")
            
            # If no existing data, add as new
            if not all_related_data:
                logger.info(f"No existing content found for {stocks}, adding as new")
                return section
                
            # Re-rank the related documents based on embedding similarity
            ranked_docs = []
            
            # Calculate similarity scores for all documents
            for doc in all_related_data:
                doc_metadata = doc.get('metadata', {})
                
                # Skip if necessary metadata is missing
                if not doc_metadata or not doc_metadata.get('text'):
                    continue
                    
                # Get document embedding
                doc_text = doc_metadata.get('text', '')
                doc_embedding = await embedding_model._aget_text_embedding(doc_text)
                
                # Calculate similarity score
                similarity = cosine_similarity(new_embedding, doc_embedding)
                
                # Parse the document date
                try:
                    doc_date = datetime.datetime.strptime(doc_metadata.get('date', ''), "%Y-%m-%d")
                except ValueError:
                    doc_date = parse_date(doc_metadata.get('date', ''))
                
                # Add to ranked list with information
                ranked_docs.append({
                    'doc': doc,
                    'similarity': similarity,
                    'date': doc_date,
                    'doc_id': doc_metadata.get('doc_id'),
                    'stocks': doc_metadata.get('stocks', [])
                })
            
            # Sort by similarity (highest first)
            ranked_docs.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Set thresholds for decision making
            high_similarity_threshold = 0.90  # Very similar content
            moderate_similarity_threshold = 0.75  # Related but with potential differences
            
            # Date of the current section
            try:
                current_date = datetime.datetime.strptime(metadata.get('date', ''), "%Y-%m-%d")
            except ValueError:
                current_date = parse_date(metadata.get('date', ''))
            
            # Log analysis of top documents
            if ranked_docs:
                top_doc = ranked_docs[0]
                logger.info(f"Top matching document similarity: {top_doc['similarity']:.3f}, " 
                          f"date: {top_doc['date']}, stocks: {top_doc['doc'].get('metadata', {}).get('stocks', [])}")
                
                # High similarity documents indicate potential duplicates
                if top_doc['similarity'] >= high_similarity_threshold:
                    # If current document is newer, it should replace old ones
                    if current_date > top_doc['date']:
                        logger.info(f"New document is more recent ({current_date} > {top_doc['date']}) "
                                  f"with high similarity ({top_doc['similarity']:.3f}). Replacing old document.")
                        
                        # Mark similar documents for deletion
                        docs_to_delete = []
                        for doc in ranked_docs:
                            if doc['similarity'] >= moderate_similarity_threshold:
                                doc_id = doc.get('doc_id')
                                if doc_id:
                                    logger.info(f"Marking document {doc_id} for deletion - replaced by newer content")
                                    docs_to_delete.append(doc_id)
                                    
                        # Delete outdated documents if we have their IDs
                        if docs_to_delete:
                            for doc_id in docs_to_delete:
                                # TODO: Implement vector store deletion
                                # await self.pinecone_tool.delete_document(doc_id)
                                logger.info(f"Deleting outdated document: {doc_id}")
                                
                        # Use the current section as the update
                        return section
                    else:
                        # Current document is older than existing, likely duplicate
                        logger.info(f"Current document is older or same date as existing. "
                                  f"Skipping to avoid duplication. Similarity: {top_doc['similarity']:.3f}")
                        return None
                        
                # Moderate similarity documents may need merging
                elif top_doc['similarity'] >= moderate_similarity_threshold:
                    # Identify candidates for merging (all with moderate similarity)
                    merge_candidates = [item for item in ranked_docs 
                                      if item['similarity'] >= moderate_similarity_threshold]
                    
                    if merge_candidates:
                        logger.info(f"Found {len(merge_candidates)} documents for potential merging "
                                  f"with similarities from {merge_candidates[-1]['similarity']:.3f} "
                                  f"to {merge_candidates[0]['similarity']:.3f}")
                        
                        # Merge content from all moderate similarity documents
                        merged_section = await self._merge_section_content(section, 
                                                                         [cand['doc'] for cand in merge_candidates])
                        
                        # Mark original documents for deletion after successful merge
                        for cand in merge_candidates:
                            doc_id = cand.get('doc_id')
                            if doc_id:
                                logger.info(f"Marking document {doc_id} for deletion after merging")
                                # TODO: Implement vector store deletion
                                # await self.pinecone_tool.delete_document(doc_id)
                        
                        return merged_section
                
            # Low similarity - treat as new content
            logger.info(f"Content for {stocks} has low similarity to existing records. Adding as new.")
            return section

        except Exception as e:
            logger.error(f"Error in advanced section processing: {str(e)}")
            traceback.print_exc()
            return section  # Return original section on error

    async def _merge_section_content(self, new_section: Dict, existing_docs: List) -> Dict:
        """
        Merge content from multiple sections to create a consolidated view.
        Prioritizes newer content while preserving unique insights.
        """
        try:
            # Create a copy of the new section as our base
            merged_section = copy.deepcopy(new_section)
            
            # Extract key data from existing documents
            existing_content = []
            for doc in existing_docs:
                try:
                    metadata = doc.get('metadata', {})
                    
                    # Skip if missing key metadata
                    if not metadata:
                        continue
                        
                    # Parse metadata into a usable format
                    doc_data = {
                        'text': metadata.get('text', ''),
                        'stocks': metadata.get('stocks', []),
                        'date': parse_date(metadata.get('date', '')),
                        'tags': metadata.get('tags', []),
                        'key_points': metadata.get('key_points', []),
                        'technical_analysis': metadata.get('technical_analysis', ''),
                        'frame_paths': metadata.get('frame_paths', [])
                    }
                    
                    existing_content.append(doc_data)
                except Exception as e:
                    logger.error(f"Error processing document during merge: {str(e)}")
            
            # Sort by date (newest first)
            existing_content.sort(key=lambda x: x['date'], reverse=True)
            
            # --- Merge key points ---
            all_key_points = set(new_section.get('key_points', []))
            for doc in existing_content:
                all_key_points.update(doc.get('key_points', []))
            merged_section['key_points'] = list(all_key_points)
            
            # --- Merge frame paths ---
            all_frames = set(new_section.get('frame_paths', []))
            for doc in existing_content:
                all_frames.update(doc.get('frame_paths', []))
            merged_section['frame_paths'] = list(all_frames)
            
            # --- Merge tags ---
            all_tags = set(new_section.get('tags', []))
            for doc in existing_content:
                all_tags.update(doc.get('tags', []))
            merged_section['tags'] = list(all_tags)
            
            # --- Merge technical analysis ---
            # Use new section's analysis but could implement more sophisticated merging
            # if needed (e.g., using LLM to consolidate multiple analyses)
            
            logger.info(f"Merged section now has {len(merged_section.get('key_points', []))} key points and "
                      f"{len(merged_section.get('frame_paths', []))} frame paths")
            
            return merged_section
            
        except Exception as e:
            logger.error(f"Error merging section content: {str(e)}")
            return new_section  # Return original on error

    def _compare_sections(self, new_section: Dict, existing_section: Dict) -> List[str]:
        """Compare two sections and return a list of changes."""
        changes = []
        
        # Compare stocks
        new_stocks = set(new_section.get("stocks", []))
        existing_stocks = set(existing_section.get("stocks", []))
        if new_stocks != existing_stocks:
            changes.append("stocks")
            
        # Compare key points
        new_points = set(new_section.get("key_points", []))
        existing_points = set(existing_section.get("key_points", []))
        if new_points != existing_points:
            changes.append("key_points")
            
        # Compare summaries
        if new_section.get("summary") != existing_section.get("summary"):
            changes.append("summary")
            
        # Compare frame paths
        new_frames = set(new_section.get("frame_paths", []))
        existing_frames = set(existing_section.get("frame_paths", []))
        if new_frames != existing_frames:
            changes.append("frames")
            
        return changes

    def _detect_content_changes(self, new_section: Dict[str, Any], existing_section: Dict[str, Any]) -> List[str]:
        """Detect meaningful changes between two sections."""
        changes = []
        
        # Compare key points
        new_points = set(new_section.get("key_points", []))
        existing_points = set(existing_section.get("key_points", []))
        if new_points != existing_points:
            changes.append("key_points")
            
        # Compare summaries
        if new_section.get("summary") != existing_section.get("summary"):
            changes.append("summary")
            
        # Compare frame paths
        new_frames = set(new_section.get("frame_paths", []))
        existing_frames = set(existing_section.get("frame_paths", []))
        if new_frames != existing_frames:
            changes.append("frames")
            
        return changes

    async def execute(self, market_report: Dict) -> Dict:
        """Execute RAG workflow with improved chunking and vectorization strategy."""
        try:
            sections_to_update = []
            processed_sections = []
            section_indices = {}  # Track original indices

            # Process each section individually
            for idx, section in enumerate(market_report.get('sections', [])):
                # Store the original index for each section
                section_key = json.dumps(section.get('stocks', []))
                section_indices[section_key] = idx
                
                processed_section = await self._process_section_v2(section, market_report, idx)
                if processed_section:
                    sections_to_update.append(processed_section)
                processed_sections.append(processed_section or section)

            if not sections_to_update:
                logger.info("No sections require updates")
                return {"status": "no_updates_needed"}

            # Prepare report for Notion update - ensure frame_paths are included
            notion_report = {
                "Date": market_report.get('Date'),
                "Channel name": market_report.get('Channel name'),
                "sections": sections_to_update
            }
            
            # Verify frames exist in the sections
            for section in notion_report["sections"]:
                if "frame_paths" in section:
                    for frame_path in section["frame_paths"]:
                        if not os.path.exists(normalize_path(frame_path)):
                            logger.warning(f"Frame path does not exist: {frame_path}")
                            # Remove non-existent frames
                            section["frame_paths"] = [f for f in section["frame_paths"] if os.path.exists(normalize_path(f))]
                    logger.info(f"Section has {len(section.get('frame_paths', []))} valid frames")
                else:
                    logger.warning(f"Section for stocks {section.get('stocks', [])} has no frames")

            # Save debug information
            debug_file = self.debug_dir / f"notion_data_{int(time.time())}.json"
            with open(debug_file, "w") as f:
                
                json.dump(notion_report, f, indent=2)
            logger.info(f"Saved data being sent to Notion: {debug_file}")

            # Update Notion
            logger.info("Calling Notion agent execute method...")
            try:
                notion_result = await self.notion_agent.execute(notion_report)
                logger.info(f"Notion update result: {notion_result}")
            except Exception as e:
                logger.error(f"Error executing Notion agent: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                notion_result = {"status": "error", "message": str(e)}

            # Update vector store with new sections
            for section in sections_to_update:
                section_text = self._prepare_section_text(section)
                # Get the original index using the stocks as a key
                section_key = json.dumps(section.get('stocks', []))
                section_idx = section_indices.get(section_key, 0)
                
                metadata = self._prepare_section_metadata(
                    section, 
                    market_report, 
                    section_idx
                )
                
                doc = Document(
                    text=section_text,
                    metadata=metadata
                )
                await self.pinecone_tool.process_documents([doc])

            return notion_report

        except Exception as e:
            logger.error(f"Error in RAG execution: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

@track_agent(name="NotionAgent")
class NotionAgent(BaseFinanceAgent):
    """Agent for updating Notion database with technical analysis data"""
    
    def __init__(self, tools: List[BaseTool], llm: LLM, memory: ChatMemoryBuffer, config: Dict[str, Any]):
        """Initialize Notion agent"""
        # Store the original config
        self.config = config.copy()
        
        # Set explicit ReAct mode configuration 
        self.config['response_mode'] = 'REACT'
        self.config['force_tool_use'] = True
        self.config['verbose'] = True
        
        # Find and log available tools
        self.notion_tool = None
        for tool in tools:
            try:
                if 'notion' in tool.metadata.name.lower():
                    self.notion_tool = tool
                    logger.info(f"Found Notion tool: {tool.metadata.name}")
                    # Log available methods for debugging
                    for method in dir(tool):
                        if callable(getattr(tool, method)) and not method.startswith('_'):
                            logger.info(f"  Tool method available: {method}")
            except Exception as e:
                logger.error(f"Error inspecting tool {tool}: {str(e)}")
        
        if not self.notion_tool:
            logger.warning("No Notion tool found in provided tools")
            # Attempt to create a notion tool as fallback
            try:
                from app.tools.notion_tool_v2 import NotionTool
                notion_api_key = os.environ.get('NOTION_API_KEY')
                notion_database_id = os.environ.get('NOTION_DATABASE_ID')
                
                if notion_api_key and notion_database_id:
                    fallback_tool = NotionTool(notion_api_key, notion_database_id)
                    tools.append(fallback_tool)
                    self.notion_tool = fallback_tool
                    logger.info("Created fallback Notion tool")
                else:
                    logger.error("Failed to create fallback Notion tool: Missing environment variables")
            except Exception as e:
                logger.error(f"Failed to create fallback Notion tool: {str(e)}")
        
        system_prompt_lines = [
            "You are a Notion Database Agent responsible for updating stock analysis data in Notion.",
            "",
            "YOUR ONLY TASK is to update Notion with stock analysis data using the notion_tool.",
            "",
            "IMPORTANT DATA STRUCTURE:",
            "The input data follows this format:",
            "{",
            "    \"Date\": \"YYYY-MM-DD\",",
            "    \"Channel name\": \"channel_name\",",
            "    \"sections\": [",
            "        {",
            "            \"topic\": \"Technical Profile: TICKER\",",
            "            \"stocks\": [\"TICKER\"],  # This is the array of stock tickers to process",
            "            \"summary\": \"Analysis summary...\",",
            "            \"key_points\": [\"Point 1\", \"Point 2\"],",
            "            \"frame_paths\": [\"/path/to/chart1.jpg\", \"/path/to/chart2.jpg\"]",
            "        },",
            "        ... more sections ...",
            "    ]",
            "}",
            "",
            "IMPORTANT:",
            "1. For each SECTION in the data, you MUST process ALL stocks listed in that section's 'stocks' array",
            "2. For each stock ticker, you MUST call notion_tool",
            "3. You MUST follow the EXACT sequence of steps for each stock:",
            "   a) Check if the stock page exists with get_stock_page",
            "   b) Update the existing page OR create a new page",
            "   c) Add each chart/image to the page",
            "",
            "ALWAYS FORMAT YOUR TOOL CALLS LIKE THIS:",
            "Action: notion_tool",
            "Action Input: {\"input\": {\"operation\": \"get_stock_page\", \"ticker\": \"AAPL\"}}",
            "",
            "APPROVED OPERATIONS:",
            "- get_stock_page: Check if a stock page exists",
            "- create_or_update_stock_page: Create or update a stock page",
            "- update_technical_analysis: Update technical analysis on a page",
            "- add_chart_to_page: Add a chart image to a page",
            "",
            "EXACT PROCESS FOR EACH STOCK:",
            "1. First check if page exists:",
            "   Action: notion_tool",
            "   Action Input: {\"input\": {\"operation\": \"get_stock_page\", \"ticker\": \"AAPL\"}}",
            "",
            "2. If page exists, update it:",
            "   Action: notion_tool",
            "   Action Input: {\"input\": {\"operation\": \"update_technical_analysis\", \"page_id\": \"page_id_here\", \"content\": \"Technical analysis content\", \"channel_name\": \"Channel\"}} ",
            "",
            "3. If page does not exist, create it:",
            "   Action: notion_tool",
            "   Action Input: {\"input\": {\"operation\": \"create_or_update_stock_page\", \"ticker\": \"AAPL\", \"content\": {\"summary\": \"Analysis\", \"key_points\": [\"Point 1\"]}}}",
            "",
            "4. Add each chart to the page:",
            "   Action: notion_tool",
            "   Action Input: {\"input\": {\"operation\": \"add_chart_to_page\", \"page_id\": \"page_id_here\", \"image_path\": \"/path/to/image.jpg\"}}",
            "",
            "YOU MUST COMPLETE ALL STEPS FOR EACH STOCK IN THE DATA.",
            "",
            "DATA TO PROCESS:",
        ]
        
        # Join the lines with newlines to create the complete system prompt
        system_prompt_main = "\n".join(system_prompt_lines)
        
        # Update the system prompt in the config with all parts
        self.config['system_prompt'] = system_prompt_main
        
        # Create a properly escaped system prompt
        escaped_system_prompt = self.config.get('system_prompt', '').replace('{', '{{').replace('}', '}}')
        
        # Provide a special tool handler that wraps the call with input
        def tool_metadata_modifier_fn(tool_metadata):
            modified_metadata = copy.deepcopy(tool_metadata)
            # Add a reminder about input parameter in the description
            if 'notion' in modified_metadata.name.lower():
                modified_metadata.description += "\nIMPORTANT: All parameters must be wrapped in an 'input' field."
            return modified_metadata
        
        # Create the ReAct agent directly instead of using BaseFinanceAgent
        self.agent = ReActAgent.from_tools(
            tools=tools,
            llm=llm,
            system_prompt=escaped_system_prompt,
            max_iterations=self.config.get('max_iterations', 15),
            verbose=True,
            function_calling=True,  # Enable function calling for proper tool usage
            tool_metadata_modifier_fn=tool_metadata_modifier_fn
        )
        
        # Register a tool callback for notion_tool to fix input format
        self.agent.callback_manager.on_tool_start = self._handle_tool_start
        
        # Store the tools and other parameters for future use
        self._tools = tools
        self._llm = llm
        self._memory = memory
        
        logger.info("NotionAgent initialized with ReAct configuration")
        
        # Create a custom execute method that handles notion_tool specially
        self._original_execute = self.agent.aquery
    
    async def _handle_tool_start(self, tool_name, tool_input):
        """Handle tool start event to fix input format for notion_tool"""
        logger.info(f"TOOL START HANDLER: {tool_name} with input type: {type(tool_input)}")
        
        if tool_name == "notion_tool" and self.notion_tool:
            logger.info(f"Tool start: {tool_name} with input: {json.dumps(tool_input, indent=2)}")
            
            # Check if input is properly structured
            if isinstance(tool_input, dict):
                # If input is a dict but doesn't have the expected structure
                if "input" not in tool_input:
                    # Wrap parameters in input field
                    logger.info(f"Wrapping tool input in 'input' field: {json.dumps(tool_input, indent=2)}")
                    wrapped_input = {"input": tool_input}
                    logger.info(f"Wrapped input: {json.dumps(wrapped_input, indent=2)}")
                    return wrapped_input
            else:
                logger.warning(f"Tool input is not a dictionary: {type(tool_input)}")
        else:
            logger.info(f"Non-notion tool or notion_tool not found: {tool_name}")
        
        # For other tools or already wrapped inputs, return as is
        return tool_input

    async def execute(self, input_text: Union[str, Dict]) -> Dict:
        # Track retry attempts
        max_retries = 3
        retry_count = 0
        last_error = None
        
        # Log debug information about the input
        if isinstance(input_text, dict):
            logger.info(f"NotionAgent received dictionary input with keys: {list(input_text.keys())}")
            
            # Extract stock data from sections
            if 'sections' in input_text:
                # Each section should have 'stocks' field with stock tickers
                all_stocks = []
                for section in input_text.get('sections', []):
                    if 'stocks' in section:
                        all_stocks.extend(section['stocks'])
                
                logger.info(f"Processing {len(all_stocks)} stocks from sections: {', '.join(all_stocks)}")
            else:
                # Legacy format with direct stocks array
                stocks_data = input_text.get('stocks', [])
                logger.info(f"Processing {len(stocks_data)} stocks: {', '.join([s.get('ticker', 'unknown') for s in stocks_data if isinstance(s, dict) and 'ticker' in s])}")
        else:
            logger.info(f"NotionAgent received text input of length: {len(input_text)}")
            
        while retry_count < max_retries:
            try:
                # Format the input data to create a clear instruction for the Notion update
                if isinstance(input_text, dict):
                    formatted_instruction = "Please update Notion with the following stock analysis data:"
                    
                    if not self.notion_tool:
                        logger.error("Notion tool not available for direct execution")
                        return {"error": "Notion tool not found", "status": "failed"}
                    
                    successful_updates = 0
                    failed_updates = 0
                    all_results = []
                    
                    # Process all sections directly
                    if 'sections' in input_text:
                        sections = input_text.get('sections', [])
                        logger.info(f"Directly processing {len(sections)} sections")
                        
                        for section in sections:
                            if 'stocks' not in section:
                                logger.warning(f"Section missing stocks: {section.get('topic', 'unknown')}")
                                continue
                                
                            stocks = section.get('stocks', [])
                            summary = section.get('summary', '')
                            key_points = section.get('key_points', [])
                            frames = section.get('frame_paths', [])
                            
                            for stock in stocks:
                                logger.info(f"Processing stock {stock}")
                                try:
                                    # Step 1: Get stock page
                                    page_result = await self.notion_tool.arun({
                                        "operation": "get_stock_page", 
                                        "ticker": stock
                                    })
                                    
                                    # Process the result to extract page_id
                                    page_id = None
                                    if isinstance(page_result, str) and "page_id" in page_result:
                                        try:
                                            page_data = json.loads(page_result)
                                            page_id = page_data.get("page_id")
                                        except json.JSONDecodeError:
                                            # Try to extract page_id from the string
                                            import re
                                            match = re.search(r'"page_id":\s*"([^"]+)"', page_result)
                                            if match:
                                                page_id = match.group(1)
                                    
                                    # Step 2: Update or create page
                                    if page_id:
                                        logger.info(f"Updating existing page for {stock}")
                                        update_result = await self.notion_tool.arun({
                                            "operation": "update_technical_analysis",
                                            "page_id": page_id,
                                            "content": summary,
                                            "channel_name": input_text.get('Channel name', 'Unknown')
                                        })
                                    else:
                                        logger.info(f"Creating new page for {stock}")
                                        create_result = await self.notion_tool.arun({
                                            "operation": "create_or_update_stock_page",
                                            "ticker": stock,
                                            "content": {
                                                "summary": summary,
                                                "key_points": key_points
                                            }
                                        })
                                        
                                        # Try to extract page_id from creation result
                                        if isinstance(create_result, str) and "page_id" in create_result:
                                            try:
                                                page_data = json.loads(create_result)
                                                page_id = page_data.get("page_id")
                                            except json.JSONDecodeError:
                                                # Try to extract page_id from the string
                                                import re
                                                match = re.search(r'"page_id":\s*"([^"]+)"', create_result)
                                                if match:
                                                    page_id = match.group(1)
                                    
                                    # Step 3: Add charts if we have a page_id
                                    if page_id and frames:
                                        for frame in frames:
                                            try:
                                                logger.info(f"Adding chart {frame} to page for {stock}")
                                                chart_result = await self.notion_tool.arun({
                                                    "operation": "add_chart_to_page",
                                                    "page_id": page_id,
                                                    "image_path": frame
                                                })
                                                all_results.append(f"Added chart to {stock}: {chart_result}")
                                            except Exception as e:
                                                logger.error(f"Error adding chart {frame} to {stock}: {str(e)}")
                                                all_results.append(f"Failed to add chart to {stock}: {str(e)}")
                                                failed_updates += 1
                                    
                                    successful_updates += 1
                                    all_results.append(f"Successfully processed {stock}")
                                
                                except Exception as e:
                                    logger.error(f"Error processing stock {stock}: {str(e)}")
                                    all_results.append(f"Failed to process {stock}: {str(e)}")
                                    failed_updates += 1
                    
                    # Return formatted results
                    result = {
                        "output": "\n".join(all_results),
                        "successful_updates": successful_updates,
                        "failed_updates": failed_updates,
                        "status": "success" if successful_updates > 0 else "failed"
                    }
                    return result
                else:
                    # For text input, just return a message
                    result = {
                        "output": "No structured data provided for Notion updates", 
                        "status": "failed"
                    }
                    return result
                
            except Exception as e:
                last_error = e
                logger.error(f"Error in direct Notion execution: {str(e)}")
                logger.error(f"Detailed traceback: {traceback.format_exc()}")
                
                retry_count += 1
                
                # Wait before retry with exponential backoff
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # 2, 4, 8 seconds
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
        
        # If we've exhausted retries, return an error
        error_message = str(last_error) if last_error else "Unknown error in NotionAgent execution"
        return {"error": error_message, "status": "failed"}

    async def process_technical_analysis(self, data: Dict[str, Any]) -> Dict:
        """Process technical analysis data and update Notion database.
        
        Args:
            data: Dictionary containing technical analysis data with sections, stocks, etc.
            
        Returns:
            Dictionary with status and results
        """
        try:
            # Validate input
            if not isinstance(data, dict):
                raise ValueError("Input data must be a dictionary")
                
            # Check for required fields
            if 'sections' not in data:
                logger.warning("Input data missing 'sections' key")
                
            # Check for notion tool
            if not self.notion_tool:
                logger.error("NotionAgent cannot operate without a Notion tool")
                return {
                    "status": "error",
                    "message": "No Notion tool available"
                }
                
            # Extract all unique stock tickers from the sections
            all_stocks = set()
            for section in data.get('sections', []):
                stocks = section.get('stocks', [])
                all_stocks.update(stocks)
                
            logger.info(f"Processing technical analysis for {len(all_stocks)} stocks: {', '.join(all_stocks)}")
            
            # Create a formatted prompt for the agent with explicit tool usage examples
            instructions = r"""
# Notion Database Update Task

You are tasked with updating a Notion database with technical analysis data for stocks. 
This task REQUIRES using the notion_tool with EXACT ReAct syntax.

## STRICT REACT FORMAT REQUIREMENTS:

Follow this EXACT FORMAT for every tool call:

```
Thought: <reasoning about what to do next>
Action: notion_tool
Action Input: {"input": {"operation": "<operation_name>", <operation_specific_parameters>}}
```

## PROCESS FOR EACH STOCK:

1. CHECK IF STOCK EXISTS:
   Thought: First I need to check if the stock page exists
   Action: notion_tool
   Action Input: {"input": {"operation": "get_stock_page", "ticker": "AAPL"}}

2. IF PAGE EXISTS:
   Thought: The page exists, I'll update its technical analysis
   Action: notion_tool
   Action Input: {"input": {"operation": "update_technical_analysis", "page_id": "page_id_here", "content": "Technical analysis content", "channel_name": "Channel"}} 

3. IF PAGE DOES NOT EXIST:
   Thought: The page doesn't exist, I'll create a new one
   Action: notion_tool
   Action Input: {"input": {"operation": "create_or_update_stock_page", "ticker": "AAPL", "content": "Example summary"}} 

4. FOR EACH FRAME:
   Thought: I need to add the chart to the page
   Action: notion_tool
   Action Input: {"input": {"operation": "add_chart_to_page", "page_id": "page_id_here", "image_path": "/path/to/image.jpg"}} 

## IMPORTANT RULES:

1. YOU MUST USE the notion_tool - no other tools will work
2. YOU MUST INCLUDE the "input" wrapper parameter containing all operation details
3. YOU MUST PROCESS EACH STOCK INDIVIDUALLY following the sequence above
4. DO NOT SKIP any steps in the process
5. NEVER format your action inputs in code blocks - use EXACT ReAct syntax shown above
6. ALWAYS wait for the response from each tool call before proceeding
7. ENSURE you process EVERY stock mentioned in the input data
8. OPERATIONS MUST match exactly: "get_stock_page", "update_technical_analysis", "create_or_update_stock_page", "add_chart_to_page"

## DATA TO PROCESS:

Process the following technical analysis data:
{{ ... }}
"""
            
            # Ensure data is properly formatted as JSON 
            json_data = json.dumps(data, indent=2)
            prompt = instructions + "\n\n" + json_data
            
            # Execute the agent
            logger.info("Running NotionAgent with detailed ReAct instructions")
            result = await self.execute(prompt)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in process_technical_analysis: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc()
            }

class AgentWorkflow:
    """Agent workflow class"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize agent workflow"""
        self.config = config
        self.agents = self._initialize_agents()
        
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize agents"""
        llm_factory = LLMFactory()
        return {
            'technical': TechnicalAnalysisAgent(
                tools=[NotionTool()],
                llm=llm_factory.create_llm(self.config.get('technical', {}).get('llm', {})),
                memory=ChatMemoryBuffer.from_defaults(),
                config=self.config.get('technical', {})
            ),
            'market': MarketAnalysisAgent(
                tools=[NotionTool()],
                llm=llm_factory.create_llm(self.config.get('market', {}).get('llm', {})),
                memory=ChatMemoryBuffer.from_defaults(),
                config=self.config.get('market', {})
            ),
            'rag': RAGAgent(
                config=self.config.get('rag', {}),
                llm_service=llm_factory.create_llm(self.config.get('rag', {}).get('llm', {}))
            ),
            'notion': NotionAgent(
                tools=[NotionTool()],
                llm=llm_factory.create_llm(self.config.get('notion', {}).get('llm', {})),
                memory=ChatMemoryBuffer.from_defaults(),
                config=self.config.get('notion', {})
            )
        }
        
    async def execute(self, data: Dict) -> Dict:
        """Execute agent workflow"""
        try:
            # Step 1: Technical Analysis
            technical_result = await self.agents['technical'].execute(data)
            
            # Step 2: Market Analysis
            market_result = await self.agents['market'].execute(technical_result)
            
            # Step 3: RAG Processing
            rag_result = await self.agents['rag'].execute(market_result)
            
            # Step 4: Notion Update
            final_result = await self.agents['notion'].execute(rag_result)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in agent workflow execution: {str(e)}")
            raise
