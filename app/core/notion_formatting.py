"""
Rich text formatting utilities for Notion integration.
"""
import logging
from typing import Dict, Any, List, Optional, Union
import re

logger = logging.getLogger(__name__)

class NotionRichTextFormatter:
    """Format analysis data into rich Notion blocks with enhanced readability"""
    
    def __init__(self):
        self.emoji_map = {
            'bullish': '📈',
            'bearish': '📉',
            'neutral': '⚖️',
            'support': '🛡️',
            'resistance': '🚧',
            'target': '🎯',
            'risk': '⚠️',
            'pattern': '📊',
            'setup': '⚙️',
            'trigger': '🔥',
            'daily': '📅',
            'weekly': '📆',
            'monthly': '🗓️'
        }
    
    def format_stock_analysis(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert stock analysis to rich Notion blocks"""
        notion_blocks = []
        
        # Check if this is a full analysis data structure or individual section
        if 'sections' in analysis_data:
            # Full analysis data - process all sections
            # Add header with channel and date info
            if 'Channel name' in analysis_data or 'channel_name' in analysis_data:
                channel_name = analysis_data.get('Channel name', analysis_data.get('channel_name', ''))
                date = analysis_data.get('Date', '')
                
                notion_blocks.append({
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [
                            {"type": "text", "text": {"content": f"📺 {channel_name}"}, "annotations": {"bold": True, "color": "blue"}}
                        ]
                    }
                })
                
                if date:
                    notion_blocks.append({
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {"type": "text", "text": {"content": f"📅 Analysis Date: {date}"}, "annotations": {"italic": True}}
                            ]
                        }
                    })
            
            # Process each stock section
            for section in analysis_data.get('sections', []):
                if 'stocks' in section and section['stocks']:
                    stock_blocks = self._format_stock_section(section)
                    notion_blocks.extend(stock_blocks)
        else:
            # Individual section - format directly
            if 'stocks' in analysis_data and analysis_data['stocks']:
                stock_blocks = self._format_stock_section(analysis_data)
                notion_blocks.extend(stock_blocks)
        
        return notion_blocks
    
    def _format_stock_section(self, section: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format individual stock section with rich text"""
        blocks = []
        symbol = section['stocks'][0] if section.get('stocks') else 'Unknown'
        
        # Stock header
        blocks.append({
            "type": "heading_2",
            "heading_2": {
                "rich_text": [
                    {"type": "text", "text": {"content": f"📈 ${symbol} Technical Analysis"}, "annotations": {"bold": True, "color": "green"}}
                ]
            }
        })
        
        # Summary section with enhanced formatting
        summary = section.get('summary', '')
        if summary:
            formatted_summary = self._format_summary_text(summary)
            blocks.append({
                "type": "paragraph",
                "paragraph": {
                    "rich_text": formatted_summary
                }
            })
        
        # Key points as structured list
        key_points = section.get('key_points', [])
        if key_points:
            blocks.append({
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": "🔍 Key Analysis Points"}, "annotations": {"bold": True}}]
                }
            })
            
            for point in key_points:
                formatted_point = self._format_key_point(point)
                blocks.append({
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": formatted_point
                    }
                })
        
        # Add preserved technical data if available
        if 'preserved_technical_data' in section:
            tech_blocks = self._format_technical_data(section['preserved_technical_data'])
            blocks.extend(tech_blocks)
        
        # Add preserved price data if available
        if 'preserved_price_data' in section:
            price_blocks = self._format_price_data(section['preserved_price_data'])
            blocks.extend(price_blocks)
        
        # Add divider
        blocks.append({
            "type": "divider",
            "divider": {}
        })
        
        return blocks
    
    def _format_summary_text(self, summary: str) -> List[Dict[str, Any]]:
        """Format summary text with emphasis and structure"""
        rich_text = []
        
        # Split summary into sentences for better formatting
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            # Add emphasis to key terms
            formatted_sentence = self._add_text_emphasis(sentence)
            rich_text.extend(formatted_sentence)
            
            # Add line break between sentences for readability
            if i < len(sentences) - 1:
                rich_text.append({"type": "text", "text": {"content": " "}})
        
        return rich_text
    
    def _add_text_emphasis(self, text: str) -> List[Dict[str, Any]]:
        """Add emphasis to key financial and technical terms"""
        rich_text = []
        
        # Key terms to emphasize
        emphasis_patterns = {
            r'\$[\d,]+\.?\d*': {'bold': True, 'color': 'green'},  # Price levels
            r'\b(?:support|resistance|breakout|breakdown)\b': {'bold': True, 'color': 'blue'},  # Technical terms
            r'\b(?:bullish|bearish|overbought|oversold)\b': {'bold': True, 'color': 'orange'},  # Sentiment
            r'\b(?:daily|weekly|monthly)\b': {'italic': True},  # Timeframes
            r'\b(?:Bollinger|Ichimoku|Fibonacci|MACD|RSI)\b': {'bold': True, 'color': 'purple'},  # Indicators
        }
        
        current_pos = 0
        matches = []
        
        # Find all matches
        for pattern, formatting in emphasis_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append((match.start(), match.end(), match.group(), formatting))
        
        # Sort matches by position
        matches.sort(key=lambda x: x[0])
        
        # Build rich text with formatting
        for start, end, matched_text, formatting in matches:
            # Add text before match
            if current_pos < start:
                rich_text.append({
                    "type": "text",
                    "text": {"content": text[current_pos:start]}
                })
            
            # Add formatted match
            rich_text.append({
                "type": "text",
                "text": {"content": matched_text},
                "annotations": formatting
            })
            
            current_pos = end
        
        # Add remaining text
        if current_pos < len(text):
            rich_text.append({
                "type": "text",
                "text": {"content": text[current_pos:]}
            })
        
        # If no matches found, return plain text
        if not matches:
            rich_text.append({
                "type": "text",
                "text": {"content": text}
            })
        
        return rich_text
    
    def _format_key_point(self, point: str) -> List[Dict[str, Any]]:
        """Format individual key point with structure and emphasis"""
        # Extract category if present (e.g., "Pattern:", "Setup:", etc.)
        category_match = re.match(r'^(\w+):\s*(.+)', point)
        
        if category_match:
            category = category_match.group(1)
            content = category_match.group(2)
            
            # Get emoji for category
            emoji = self.emoji_map.get(category.lower(), '•')
            
            return [
                {"type": "text", "text": {"content": f"{emoji} {category}: "}, "annotations": {"bold": True, "color": "blue"}},
                *self._add_text_emphasis(content)
            ]
        else:
            # No category, format as regular point
            return [
                {"type": "text", "text": {"content": "• "}},
                *self._add_text_emphasis(point)
            ]
    
    def _format_technical_data(self, tech_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format preserved technical data"""
        blocks = []
        
        if tech_data.get('indicators'):
            blocks.append({
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": "📊 Technical Indicators"}, "annotations": {"bold": True}}]
                }
            })
            
            indicators_text = ", ".join(tech_data['indicators'])
            blocks.append({
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {"type": "text", "text": {"content": indicators_text}, "annotations": {"italic": True, "color": "purple"}}
                    ]
                }
            })
        
        if tech_data.get('patterns'):
            blocks.append({
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": "📈 Chart Patterns"}, "annotations": {"bold": True}}]
                }
            })
            
            for pattern in tech_data['patterns']:
                blocks.append({
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [
                            {"type": "text", "text": {"content": pattern.title()}, "annotations": {"bold": True}}
                        ]
                    }
                })
        
        return blocks
    
    def _format_price_data(self, price_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format preserved price level data"""
        blocks = []
        
        blocks.append({
            "type": "heading_3",
            "heading_3": {
                "rich_text": [{"type": "text", "text": {"content": "💰 Key Price Levels"}, "annotations": {"bold": True}}]
            }
        })
        
        # Support levels
        if price_data.get('support_levels'):
            support_text = ", ".join([f"${level}" for level in price_data['support_levels']])
            blocks.append({
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [
                        {"type": "text", "text": {"content": "🛡️ Support: "}, "annotations": {"bold": True, "color": "green"}},
                        {"type": "text", "text": {"content": support_text}, "annotations": {"bold": True}}
                    ]
                }
            })
        
        # Resistance levels
        if price_data.get('resistance_levels'):
            resistance_text = ", ".join([f"${level}" for level in price_data['resistance_levels']])
            blocks.append({
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [
                        {"type": "text", "text": {"content": "🚧 Resistance: "}, "annotations": {"bold": True, "color": "red"}},
                        {"type": "text", "text": {"content": resistance_text}, "annotations": {"bold": True}}
                    ]
                }
            })
        
        # Target levels
        if price_data.get('target_levels'):
            target_text = ", ".join([f"${level}" for level in price_data['target_levels']])
            blocks.append({
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [
                        {"type": "text", "text": {"content": "🎯 Targets: "}, "annotations": {"bold": True, "color": "blue"}},
                        {"type": "text", "text": {"content": target_text}, "annotations": {"bold": True}}
                    ]
                }
            })
        
        return blocks
    
    def create_summary_block(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary block for the entire analysis"""
        sections = analysis_data.get('sections', [])
        stock_count = len(sections)
        stocks = [section['stocks'][0] for section in sections if section.get('stocks')]
        
        summary_text = f"Analysis of {stock_count} stock{'s' if stock_count != 1 else ''}: {', '.join(stocks)}"
        
        return {
            "type": "callout",
            "callout": {
                "rich_text": [
                    {"type": "text", "text": {"content": summary_text}, "annotations": {"bold": True}}
                ],
                "icon": {"emoji": "📊"},
                "color": "blue_background"
            }
        }

class NotionBlockBuilder:
    """Helper class to build Notion blocks programmatically"""
    
    @staticmethod
    def heading(level: int, text: str, color: str = "default") -> Dict[str, Any]:
        """Create a heading block"""
        heading_type = f"heading_{level}"
        return {
            "type": heading_type,
            heading_type: {
                "rich_text": [
                    {"type": "text", "text": {"content": text}, "annotations": {"bold": True, "color": color}}
                ]
            }
        }
    
    @staticmethod
    def paragraph(text: str, formatting: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a paragraph block"""
        annotations = formatting or {}
        return {
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {"type": "text", "text": {"content": text}, "annotations": annotations}
                ]
            }
        }
    
    @staticmethod
    def bullet_point(text: str, formatting: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a bulleted list item"""
        annotations = formatting or {}
        return {
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [
                    {"type": "text", "text": {"content": text}, "annotations": annotations}
                ]
            }
        }
    
    @staticmethod
    def callout(text: str, emoji: str = "💡", color: str = "gray_background") -> Dict[str, Any]:
        """Create a callout block"""
        return {
            "type": "callout",
            "callout": {
                "rich_text": [
                    {"type": "text", "text": {"content": text}}
                ],
                "icon": {"emoji": emoji},
                "color": color
            }
        }
    
    @staticmethod
    def divider() -> Dict[str, Any]:
        """Create a divider block"""
        return {
            "type": "divider",
            "divider": {}
        }
