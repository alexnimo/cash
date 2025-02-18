"""Agent configuration module"""
from typing import Dict, Any

AGENT_CONFIG: Dict[str, Any] = {
    'debug_dir': 'debug',  # Directory for agent debug outputs
    
    'embedding': {
        'provider': 'huggingface',  # Using HuggingFace for embeddings
        'model': 'nomic-ai/nomic-embed-text-v2-moe',  # Nomic.ai's embedding model
        'dimension': 768,  # Default dimension for nomic-embed-text-v2-moe
        'metric': 'cosine',  # Distance metric
        'metadata_fields': ['source', 'date', 'stocks']  # Fields to be indexed
    },
    
    'pinecone': {
        'index_name': 'technical-analysis',  # Default Pinecone index name
        'cloud': 'aws',
        'region': 'us-east-1'
        # dimension and metric inherited from embedding config
    },
    
    'rag': {
        'llm': {
            'provider': 'gemini',  # One of: openai, gemini, sambanova, openai_like
            'model': 'models/gemini-2.0-flash-thinking-exp-1219',  # Default Gemini model
            'temperature': 0.7,
            'max_tokens': 32000,
            # Optional provider-specific settings
            'api_key': None,  # Set in environment variables
            'api_base': None,  # For openai_like
            'endpoint_url': None,  # For sambanova
            'additional_kwargs': {}
        },
        'max_iterations': 3,
        'similarity_top_k': 3,  # Number of similar documents to retrieve
        'batch_size': 10,
        'pinecone_index': 'technical-analysis',  # Name of the Pinecone index
        'pinecone_environment': 'gcp-starter',  # Pinecone environment
        'system_prompt': """You are a RAG specialist responsible for managing financial analysis data flow."""
    },
    
    'notion': {
        'llm': {
            'provider': 'gemini',
            'model': 'models/gemini-1.5-flash',
            'temperature': 0.7,
            'max_tokens': 128000
        },
        'max_iterations': 3,
        'page_structure': {
            'database_id': '',  # Set this in environment variables
            'properties': {
                'Name': 'title',
                'Status': 'select',
                'Type': 'select',
                'Date': 'date',
                'Summary': 'rich_text',
                'Key Points': 'rich_text',
                'Stocks': 'multi_select',
                'Source': 'select'
            },
            'technical_analysis': ['summary', 'key_points', 'charts'],
            'market_insights': ['summary', 'key_points', 'charts']
        },
        'system_prompt': """You are a Notion Database Expert responsible for creating and updating structured financial analysis content."""
    },
    
    'technical_analysis': {
        'llm': {
            'provider': 'gemini',
            'model': 'models/gemini-1.5-flash',
            'temperature': 0.7,
            'max_tokens': 128000
        },
        'max_iterations': 3,
        'system_prompt': """You are an expert Technical Analysis Agent specializing in stock market analysis.
        Your role is to analyze technical indicators, chart patterns, and market data to provide institutional-grade 
        technical analysis summaries.

        Core Responsibilities:
        1. Load and parse JSON analysis files
        2. Filter analysis to include only tracked stocks from Notion database
        3. Generate comprehensive technical analysis reports
        4. Ensure proper JSON formatting throughout the process

        For any input JSON file:
        1. First, retrieve the list of tracked stocks from Notion
        2. Filter the analysis to include only tracked stocks
        3. Generate a structured analysis report with the following format:

        {
            "Date": "YYYY-MM-DD",
            "Channel_name": "Technical Analysis",
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
        }

        Always ensure that:
        1. Only tracked stocks are included in the analysis
        2. All JSON formatting is handled within your responses
        3. Frame paths and technical indicators are preserved
        4. Analysis maintains institutional-grade quality"""
    },
    
    'market_analysis': {
        'llm': {
            'provider': 'gemini',
            'model': 'models/gemini-1.5-flash',
            'temperature': 0.7,
            'max_tokens': 128000
        },
        'max_iterations': 3,
        'system_prompt': """You are an expert Market Analysis Agent specializing in analyzing market trends and conditions.
        Your role is to analyze market sectors, indices, and macroeconomic factors to provide comprehensive market insights.
        
        Core Responsibilities:
        1. Monitor global market conditions
        2. Track sector rotations and performance
        3. Analyze market breadth and depth
        4. Evaluate institutional money flow
        5. Assess market sentiment indicators
        
        For each analysis:
        1. Consider multiple timeframes
        2. Cross-reference different data sources
        3. Validate findings with technical indicators
        4. Highlight potential market risks
        5. Identify emerging opportunities"""
    }
}
