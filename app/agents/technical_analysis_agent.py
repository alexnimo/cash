from typing import Dict, Any, Optional, List, Tuple
import json
import logging
import time
from pathlib import Path
import google.generativeai as genai
from app.core.settings import get_settings
from app.tools.notion_tool import NotionTool
from app.utils.langtrace_utils import get_langtrace, trace_llm_call, init_langtrace

logger = logging.getLogger(__name__)


class TechnicalAnalysisAgent:
    def __init__(self):
        """Initialize the TechnicalAnalysisAgent."""
        self.settings = get_settings()
        self.notion = NotionTool()
        
        # Initialize model
        model_name = self.settings.agents.llm.name
        genai.configure(api_key=self.settings.api.gemini_api_key)  # Use API key from api settings
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                candidate_count=1
            )
        )
        
        # Define tool implementations
        self._tool_implementations = {
            "update_notion_page": self._update_notion_page,
            "analyze_stock_charts": self._analyze_stock_charts
        }
        
    async def _get_tracked_stocks(self) -> List[str]:
        """Get list of tracked stocks from Notion DB."""
        try:
            stocks = await self.notion.get_all_tickers()
            logger.info(f"Found {len(stocks)} tracked stocks in Notion DB")
            logger.info(f"Tracked stocks: {', '.join(stocks)}")
            return stocks
        except Exception as e:
            logger.error(f"Error getting tracked stocks: {str(e)}")
            raise
            
    async def _consolidate_sections(self, sections: List[Dict]) -> Dict[str, Dict]:
        """Consolidate multiple sections of the same stock into one comprehensive analysis."""
        consolidated = {}
        
        for section in sections:
            stock = section.get("stock")
            if not stock:
                continue
                
            if stock not in consolidated:
                consolidated[stock] = {
                    "stock": stock,
                    "distilled_section": {
                        "stocks": [stock],
                        "frame_paths": [],
                        "source": "Consolidated Analysis",
                        "summary": "",
                        "key_points": []
                    },
                    "timestamp": section.get("timestamp", ""),
                    "frames_analysis": []
                }
            
            curr = consolidated[stock]
            dist_section = section.get("distilled_section", {})
            
            # Collect frame paths
            frame_paths = dist_section.get("frame_paths", [])
            curr["distilled_section"]["frame_paths"].extend(frame_paths)
            
            # Append summary
            summary = dist_section.get("summary", "").strip()
            if summary:
                if curr["distilled_section"]["summary"]:
                    curr["distilled_section"]["summary"] += "\n\n" + summary
                else:
                    curr["distilled_section"]["summary"] = summary
            
            # Collect unique key points
            key_points = dist_section.get("key_points", [])
            curr["distilled_section"]["key_points"].extend(key_points)
            
            # Store frame analysis if available
            if frame_paths:
                curr["frames_analysis"].append({
                    "paths": frame_paths,
                    "analysis": section.get("analysis", "")
                })
        
        return consolidated

    async def _analyze_frames(self, frames_analysis: List[Dict]) -> Tuple[List[str], str]:
        """Analyze multiple frame analyses to select the most relevant technical charts."""
        if not frames_analysis:
            return [], "No technical analysis charts available."
            
        prompt = """Analyze these technical chart descriptions and select the most relevant ones for technical analysis.
        Only select charts that show clear technical patterns, indicators, or signals.
        Ignore charts that don't contain technical analysis elements.
        
        Chart Analyses:
        {}
        
        Provide your response in JSON format:
        {{
            "selected_frames": ["path1", "path2"],
            "analysis_summary": "Brief technical analysis summary"
        }}
        """.format(json.dumps(frames_analysis, indent=2))
        
        try:
            response = self.model.generate_content(prompt)
            if not response or not response.text:
                return [], "No technical analysis charts selected."
                
            result = json.loads(response.text)
            return result.get("selected_frames", []), result.get("analysis_summary", "")
        except Exception as e:
            logger.error(f"Error analyzing frames: {str(e)}")
            return [], "Error analyzing technical charts."

    async def _distill_report(self, analysis_data: Dict) -> Dict:
        """Use LLM to distill and filter the stock analysis report."""
        try:
            # Get tracked stocks
            tracked_stocks = await self._get_tracked_stocks()
            if not tracked_stocks:
                logger.warning("No tracked stocks found in Notion DB")
                return {"sections": []}
                
            # Build distillation prompt
            prompt = f"""Act as a Stock Analysis Consolidator. Transform fragmented stock commentary into institutional-grade technical summaries using this strict protocol:

                    **Core Objective**  
                    Create unified technical profiles for tracked stocks by synthesizing all mentions across source materials.

                    **Tracked Stocks**  
                    {', '.join(tracked_stocks)}

                    Input Report:
                    {json.dumps(analysis_data, indent=2)}

                    **Input Processing Rules**  
                    1. Filter mercilessly - EXCLUDE all non-tracked stocks  
                    2. Preserve ALL visual references (frame_paths)  
                    3. Maintain chronological event sequence
                    4. Don not include {', '.join(tracked_stocks)} sections in case the tracked stock is not mentioned in the provided input report

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
                    ```json
                    {{
                    "Date": "",
                    "Channel name": "",
                    "sections": [
                        {{
                        "topic": "Technical Profile: {{STOCK}}",
                        "stocks": ["{{STOCK}}"],
                        "frame_paths": ["path1", ...],  
                        "source": "Composite Analysis",
                        "summary": "[Price Context] + [Volatility Profile] + [Key Risk/Return Ratio]",
                        "key_points": [
                            "Pattern: {{HistoricalPriceBehavior}}", 
                            "Trigger: {{Catalyst}}",
                            "Risk: {{StructuralWeakness}}",
                            "Level: {{CriticalPriceThreshold}}"
                        ]
                        }}
                    ]
                    }}
            """
            
            # Generate distilled report
            response = self.model.generate_content(prompt)
            
            try:
                # Log the raw response for debugging
                logger.info(f"Raw Gemini response: {response.text}")
                
                # Find the JSON part of the response
                response_text = response.text
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                
                if start == -1 or end == 0:
                    logger.error("No JSON found in response")
                    return {"sections": []}
                
                json_str = response_text[start:end]
                logger.info(f"Extracted JSON: {json_str}")
                
                # Parse and return the response
                distilled_report = json.loads(json_str)
                
                # Save debug file
                debug_dir = Path("debug")
                debug_dir.mkdir(exist_ok=True)
                debug_file = debug_dir / f"consolidated_summary_{int(time.time())}.json"
                with open(debug_file, "w") as f:
                    json.dump(distilled_report, f, indent=2)
                
                return distilled_report
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing distilled report: {str(e)}")
                return {"sections": []}
                
        except Exception as e:
            logger.error(f"Error distilling report: {str(e)}")
            raise
            
    async def _analyze_stock_charts(self, summary_data: Dict) -> str:
        """Analyze stock charts and select the best images."""
        try:
            # Upload frames if they exist
            frame_paths = summary_data.get('frame_paths', [])
            if not frame_paths:
                logger.warning("No frame paths found in summary data")
                return "No chart images available for analysis."
                
            # Log frame paths for debugging
            logger.info(f"Analyzing frames: {frame_paths}")
            
            uploaded_frames = []
            for frame_path in frame_paths:
                try:
                    file = genai.upload_file(frame_path, mime_type="image/jpeg")
                    uploaded_frames.append(file)
                    logger.info(f"Uploaded frame: {frame_path}")
                except Exception as e:
                    logger.error(f"Failed to upload frame {frame_path}: {str(e)}")
                    continue
                    
            if not uploaded_frames:
                logger.warning("Failed to upload any frames")
                return "Unable to analyze chart images due to upload errors."
            
            # Build prompt with image context
            prompt = f"""Analyze these technical analysis sections and their chart images.
            Focus on identifying key technical patterns, support/resistance levels, and trading signals.
            
            Stock: {summary_data.get('stocks', ['Unknown'])[0]}
            Summary: {summary_data.get('summary', '')}
            Key Points: {json.dumps(summary_data.get('key_points', []), indent=2)}
            
            Provide a concise technical analysis that includes:
            1. Chart patterns and formations
            2. Support and resistance levels
            3. Technical indicators (RSI, MACD, etc.)
            4. Volume analysis
            5. Trend analysis
            6. Trading signals or setups
            
            Important:
            - Keep your response under 1500 characters to fit within Notion's limits
            - Focus on the most important patterns and signals
            - Use bullet points for better readability
            - Format the analysis in markdown
            """
            
            generation_config = genai.types.GenerationConfig(
                temperature=0.1,
                candidate_count=1,
                max_output_tokens=1500  # Limit output tokens
            )
            
            # Generate content with images
            response = self.model.generate_content(
                [prompt, *uploaded_frames],
                generation_config=generation_config
            )
            
            return response.text if response and response.text else ""
            
        except Exception as e:
            logger.error(f"Error analyzing stock charts: {str(e)}")
            raise
            
    async def _update_notion_page(self, stock_symbol: str, content: str, channel_name: str, image_paths: List[str] = None) -> bool:
        """Update a Notion page with the given content."""
        try:
            print(f"\n[DEBUG] Updating Notion page for {stock_symbol}")
            print(f"[DEBUG] Image paths to process: {image_paths}")
            
            # Get existing page
            page = await self.notion.get_stock_page(stock_symbol)
            if not page:
                logger.error(f"No Notion page found for stock: {stock_symbol}")
                return False
                
            print(f"[DEBUG] Got page ID: {page['id']}")
            
            # Add each chart to the page
            if image_paths:
                for image_path in image_paths:
                    print(f"[DEBUG] Adding chart to page: {image_path}")
                    await self.notion.add_chart_to_page(page['id'], image_path)
            
            # Update the summary
            await self.notion.update_technical_analysis(
                page['id'],
                content,
                channel_name
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating Notion page: {str(e)}")
            return False

    async def execute(self, analysis_file: str):
        """Execute technical analysis from a JSON file."""
        try:
            print(f"\n[DEBUG] Starting analysis of file: {analysis_file}")
            # Load and validate the analysis file
            with open(analysis_file, 'r') as f:
                analysis_data = json.load(f)
                
            # Create debug directory if it doesn't exist
            debug_dir = Path("debug")
            debug_dir.mkdir(exist_ok=True)
            
            # Save original analysis data
            debug_file = debug_dir / f"original_analysis_{int(time.time())}.json"
            with open(debug_file, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            print(f"[DEBUG] Saved analysis to {debug_file}")
            
            # Get distilled report
            distilled_report = await self._distill_report(analysis_data)
            
            # Update Notion for each section
            for section in distilled_report.get('sections', []):
                try:
                    if section.get('summary', '').strip():
                        # Get frame paths and ensure they're absolute
                        frame_paths = section.get('frame_paths', [])
                        if frame_paths:
                            print(f"\n[DEBUG] Found frames for {section['stocks'][0]}:")
                            for path in frame_paths:
                                print(f"[DEBUG] Frame path: {path}")
                                if not Path(path).exists():
                                    print(f"[DEBUG] Frame does not exist: {path}")
                                else:
                                    print(f"[DEBUG] Frame exists: {path}")
                            
                        # Update Notion with content and frames
                        result = await self._update_notion_page(
                            stock_symbol=section['stocks'][0],  # First stock in the list
                            content=section['summary'],
                            channel_name=distilled_report.get("Channel name", "Unknown Channel"),
                            image_paths=frame_paths
                        )
                        print(f"[DEBUG] Notion update result for {section['stocks'][0]}: {result}")
                except Exception as e:
                    logger.error(f"Error updating Notion for section: {str(e)}")
                    continue
                    
            return distilled_report
            
        except Exception as e:
            logger.error(f"Error executing technical analysis: {str(e)}")
            raise
