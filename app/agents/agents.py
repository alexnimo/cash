from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
import json
import time
import asyncio
import uuid
import os
from datetime import datetime

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

from app.tools import notion_tool_v2
from app.config.agent_config import AGENT_CONFIG
from app.llm import LLMFactory

logger = logging.getLogger(__name__)

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
        self.notion_tool = next(t for t in tools if isinstance(t, notion_tool_v2.NotionAdvancedToolSpec))
        
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
        ✓ Price history markers (e.g., "$0.7→$10 post-announcement")  
        ✓ Volatility catalysts (policy changes, earnings, M&A rumors)  
        ✓ Liquidity signals (volume patterns, float analysis)  
        ✓ Structural levels (IPO price, historical support/resistance analogs)  
        ✓ Risk multipliers (dilution potential, short interest cues)  

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
            tracked_stocks_str = ", ".join(tracked_stocks)
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
            prompt = f"""Act as a Technical Analysis Consolidator. Transform this analysis data into a structured technical summary.
            Your response must be a valid JSON object with the following structure:
            {{
                "Date": "<current_date>",
                "Channel name": "<channel_name>",
                "sections": [
                    {{
                        "topic": "Technical Profile: <STOCK>",
                        "stocks": ["<STOCK>"],
                        "frame_paths": ["path1", ...],
                        "source": "Composite Analysis",
                        "summary": "<analysis_summary>",
                        "key_points": [
                            "Pattern: <pattern>",
                            "Trigger: <trigger>",
                            "Risk: <risk>",
                            "Level: <level>"
                        ]
                    }}
                ]
            }}

            **Tracked Stocks to Include**
            {tracked_stocks_str}

            **Analysis Data Content**
            {market_content}

            Follow these strict rules:
            1. ONLY include analysis for the tracked stocks listed above
            2. Maintain ALL frame_paths references
            3. Follow the exact JSON output structure shown above
            4. Ensure chronological sequence of events
            5. Extract all technical parameters as specified
            6. Construct complete technical profiles with all required elements
            7. Your response must be a valid JSON object, nothing else

            Generate the consolidated technical analysis report now."""

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
            logger.debug(f"Raw LLM response: {response_text}")
            
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
        self.notion_tool = next(t for t in tools if isinstance(t, notion_tool_v2.NotionAdvancedToolSpec))
        
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
        ✓ Price history markers (e.g., "$0.7→$10 post-announcement")  
        ✓ Volatility catalysts (policy changes, earnings, M&A rumors)  
        ✓ Liquidity signals (volume patterns, float analysis)  
        ✓ Structural levels (IPO price, historical support/resistance analogs)  
        ✓ Risk multipliers (dilution potential, short interest cues)  

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
            tracked_stocks_str = ", ".join(tracked_stocks)
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

class RAGAgent:
    """Agent for RAG operations with data verification and enrichment"""
    
    def __init__(self, config: Dict[str, Any], llm_service: Any):
        self.config = config
        self.llm_service = llm_service
        self.debug_dir = Path(AGENT_CONFIG['debug_dir'])
        self.debug_dir.mkdir(exist_ok=True)
        
        # Initialize vector store and embedding model
        self._init_vector_store()
        
        # Initialize Notion agent for updates
        self.notion_agent = NotionAgent(
            tools=[notion_tool_v2.NotionAdvancedToolSpec()],
            llm=llm_service,
            memory=ChatMemoryBuffer.from_defaults(),
            config=AGENT_CONFIG.get('notion', {})
        )

        self.system_prompt = """You are a RAG Agent specializing in enriching stock analysis reports with historical context.

        **Core Responsibilities**
        1. Compare new analysis with historical data
        2. Identify significant changes and patterns
        3. Enrich analysis with relevant historical context
        4. Maintain data quality and consistency

        **Processing Rules**
        1. Preserve original report structure
        2. Add historical context where relevant
        3. Highlight significant changes
        4. Maintain chronological accuracy

        Your task is to:
        1. Review the new market analysis
        2. Compare with historical data
        3. Enrich the analysis with relevant context
        4. Prepare the enriched report for Notion update"""
        
    def _init_vector_store(self):
        """Initialize vector store and embedding components"""
        # Initialize Pinecone tool (which will handle dimension adaptation)
        from app.tools.pinecone_tool_v2 import PineconeAdvancedToolSpec
        
        try:
            # Initialize Pinecone tool with configuration from agent_config
            self.pinecone_tool = PineconeAdvancedToolSpec()
            
            # Get index name from config
            index_name = self.config.get('pinecone_index', 'technical-analysis')
            
            # Initialize vector store with the Pinecone index
            self.vector_store = PineconeVectorStore(
                api_key=self.config.get('pinecone_api_key'),
                environment=self.config.get('pinecone_environment', 'gcp-starter'),
                index_name=index_name,
                embedding_model=self.pinecone_tool.get_embedding_model()
            )
            
            logger.info(f"Successfully initialized vector store with index: {index_name}")
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    async def execute(self, market_report: Dict) -> Dict:
        """Execute RAG workflow and update Notion"""
        try:
            # Convert report to document
            report_str = json.dumps(market_report, indent=2)
            document = Document(text=report_str, metadata={
                'type': 'technical_analysis',
                'date': market_report.get('Date'),
                'source': 'market_report'
            })
            
            # Process document through pipeline
            await self.pinecone_tool.process_documents([document])
            
            # Query similar reports
            similar_docs = await self.pinecone_tool.query_similar(report_str, top_k=5)
            
            # Create enrichment prompt
            historical_context = "\n".join([
                f"Historical Report {i+1}:\n{doc.get('content', '')}"
                for i, doc in enumerate(similar_docs)
            ])
            
            prompt = f"""Enrich this technical analysis report with historical context.

            **Current Report**
            {report_str}

            **Historical Context**
            {historical_context if similar_docs else "No historical data available."}

            Follow these rules:
            1. Compare current analysis with historical data
            2. Identify significant changes in technical patterns
            3. Add relevant historical context to each stock's analysis
            4. Maintain the exact same JSON structure
            5. Your response must be a valid JSON object

            Generate the enriched technical analysis report now."""

            # Generate enriched report using LLM
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
            logger.debug(f"Raw LLM response: {response_text}")
            
            try:
                # Try to find JSON in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start == -1 or json_end == 0:
                    logger.error("No JSON found in LLM response")
                    raise ValueError("LLM response does not contain valid JSON")
                    
                json_str = response_text[json_start:json_end]
                enriched_report = json.loads(json_str)
                
                # Process enriched report as a new document
                enriched_doc = Document(
                    text=json.dumps(enriched_report, indent=2),
                    metadata={
                        'type': 'technical_analysis',
                        'date': enriched_report.get('Date'),
                        'source': 'enriched_report'
                    }
                )
                await self.pinecone_tool.process_documents([enriched_doc])
                
                return enriched_report
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
                raise ValueError(f"Invalid JSON in LLM response: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing LLM response: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Error in RAG execution: {str(e)}")
            raise

class NotionAgent(BaseFinanceAgent):
    """Agent for Notion operations"""
    def __init__(self, tools: List[BaseTool], llm: LLM, memory: ChatMemoryBuffer, config: Dict[str, Any]):
        """Initialize Notion agent"""
        # Set system prompt
        config['system_prompt'] = """You are an expert Notion Database Agent specializing in organizing and managing financial analysis data.
        Your role is to maintain a structured database of market research, technical analysis, and trading insights.
        
        Always:
        1. Maintain consistent formatting
        2. Organize data hierarchically
        3. Tag and categorize entries appropriately
        4. Ensure data integrity and accuracy"""
        
        # Initialize base agent
        super().__init__(tools=tools, llm=llm, memory=memory, config=config)
        self.llm_service = llm  # Store LLM as llm_service
        self.notion_tool = next(t for t in tools if isinstance(t, notion_tool_v2.NotionAdvancedToolSpec))
        
    async def execute(self, input_data: Dict) -> Dict:
        """Execute Notion workflow"""
        try:
            # Ensure we have ticker information
            if not input_data.get('ticker'):
                logger.error("No ticker found in input data")
                raise ValueError("Missing ticker information")
                
            # Convert input data to string for LLM
            input_str = json.dumps(input_data, indent=2)
            
            prompt = f"""Process this analysis data and prepare it for Notion database entry.
            Format the data according to our database schema and ensure all required fields are populated.
            
            Input Data:
            {input_str}
            
            Generate a structured response that can be used to update the Notion database."""
            
            # Use LLM's generate_text or complete method
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
            if hasattr(response, 'text'):
                result = response.text
            else:
                result = str(response)
                
            # Create Notion data
            notion_data = {
                'ticker': input_data['ticker'],
                'content': {
                    'title': input_data['content']['title'],
                    'summary': input_data['content']['summary'],
                    'analysis': input_data['content']['analysis']
                }
            }
            
            # Update Notion database
            notion_result = await self.notion_tool.create_or_update_stock_page(notion_data)
            
            return {
                'input': input_data,
                'processed': notion_data,
                'notion_update': notion_result
            }
            
        except Exception as e:
            logger.error(f"Error in Notion execution: {str(e)}")
            raise

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
                tools=[notion_tool_v2.NotionAdvancedToolSpec()],
                llm=llm_factory.create_llm(self.config.get('technical', {}).get('llm', {})),
                memory=ChatMemoryBuffer.from_defaults(),
                config=self.config.get('technical', {})
            ),
            'market': MarketAnalysisAgent(
                tools=[notion_tool_v2.NotionAdvancedToolSpec()],
                llm=llm_factory.create_llm(self.config.get('market', {}).get('llm', {})),
                memory=ChatMemoryBuffer.from_defaults(),
                config=self.config.get('market', {})
            ),
            'rag': RAGAgent(
                config=self.config.get('rag', {}),
                llm_service=llm_factory.create_llm(self.config.get('rag', {}).get('llm', {}))
            ),
            'notion': NotionAgent(
                tools=[notion_tool_v2.NotionAdvancedToolSpec()],
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
