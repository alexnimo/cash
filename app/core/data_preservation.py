"""
Data preservation utilities to prevent information loss during agent processing.
"""
import logging
from typing import Dict, Any, List, Optional, Union
import json
from pathlib import Path
import re

logger = logging.getLogger(__name__)

class DataPreservationMixin:
    """Mixin class to preserve critical data during agent processing"""
    
    CRITICAL_FIELDS = [
        'specific_price_levels', 'time_ranges', 'frame_paths',
        'technical_indicators', 'support_resistance_levels',
        'start_time', 'end_time', 'stocks', 'channel_name',
        'Channel name', 'Date'
    ]
    
    def preserve_critical_data(self, original_data: Dict[str, Any], processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure critical data points are never lost during processing"""
        preserved_data = processed_data.copy()
        
        # Preserve top-level critical fields
        for field in self.CRITICAL_FIELDS:
            if field in original_data and field not in preserved_data:
                preserved_data[field] = original_data[field]
                logger.debug(f"Preserved top-level field: {field}")
        
        # Preserve section-level critical data
        if 'sections' in original_data and 'sections' in preserved_data:
            preserved_data['sections'] = self._preserve_section_data(
                original_data['sections'], 
                preserved_data['sections']
            )
        
        return preserved_data
    
    def _preserve_section_data(self, original_sections: List[Dict], processed_sections: List[Dict]) -> List[Dict]:
        """Preserve critical data within sections"""
        preserved_sections = []
        
        # Create a mapping of processed sections by stock symbol for easier matching
        processed_by_stock = {}
        for section in processed_sections:
            if 'stocks' in section and section['stocks']:
                stock = section['stocks'][0].upper()
                processed_by_stock[stock] = section
        
        # Match original sections with processed ones and preserve data
        for orig_section in original_sections:
            if 'stocks' in orig_section and orig_section['stocks']:
                stock = orig_section['stocks'][0].upper()
                
                if stock in processed_by_stock:
                    # Merge original critical data into processed section
                    preserved_section = processed_by_stock[stock].copy()
                    
                    # Preserve critical fields from original
                    for field in self.CRITICAL_FIELDS:
                        if field in orig_section and field not in preserved_section:
                            preserved_section[field] = orig_section[field]
                    
                    # Preserve detailed price information
                    preserved_section = self._preserve_price_data(orig_section, preserved_section)
                    
                    preserved_sections.append(preserved_section)
                else:
                    # If no processed version exists, keep original
                    preserved_sections.append(orig_section)
        
        # Add any processed sections that weren't matched
        for stock, section in processed_by_stock.items():
            if not any(s.get('stocks', [{}])[0].upper() == stock for s in preserved_sections):
                preserved_sections.append(section)
        
        return preserved_sections
    
    def _preserve_price_data(self, original: Dict[str, Any], processed: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and preserve specific price levels and technical data"""
        preserved = processed.copy()
        
        # Extract price levels from original summary and key_points
        price_data = self._extract_price_levels(original)
        if price_data:
            preserved['preserved_price_data'] = price_data
        
        # Preserve technical indicators
        technical_data = self._extract_technical_indicators(original)
        if technical_data:
            preserved['preserved_technical_data'] = technical_data
        
        return preserved
    
    def _extract_price_levels(self, section: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract specific price levels from text"""
        price_data = {
            'support_levels': [],
            'resistance_levels': [],
            'target_levels': [],
            'current_price': None
        }
        
        # Combine all text content for analysis
        text_content = ""
        if 'summary' in section:
            text_content += section['summary'] + " "
        if 'key_points' in section:
            if isinstance(section['key_points'], list):
                text_content += " ".join(section['key_points'])
            else:
                text_content += str(section['key_points'])
        
        # Extract price patterns (e.g., $3.53, $26, 3.89)
        price_pattern = r'\$?(\d+\.?\d*)'
        prices = re.findall(price_pattern, text_content)
        
        # Extract specific level types
        support_pattern = r'support.*?\$?(\d+\.?\d*)'
        resistance_pattern = r'resistance.*?\$?(\d+\.?\d*)'
        target_pattern = r'target.*?\$?(\d+\.?\d*)'
        
        price_data['support_levels'] = list(set(re.findall(support_pattern, text_content, re.IGNORECASE)))
        price_data['resistance_levels'] = list(set(re.findall(resistance_pattern, text_content, re.IGNORECASE)))
        price_data['target_levels'] = list(set(re.findall(target_pattern, text_content, re.IGNORECASE)))
        
        return price_data if any(price_data.values()) else {}
    
    def _extract_technical_indicators(self, section: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract technical indicators mentioned in the analysis"""
        technical_data = {
            'indicators': [],
            'patterns': [],
            'timeframes': []
        }
        
        text_content = ""
        if 'summary' in section:
            text_content += section['summary'] + " "
        if 'key_points' in section:
            if isinstance(section['key_points'], list):
                text_content += " ".join(section['key_points'])
        
        # Common technical indicators
        indicators = [
            'Bollinger Band', 'Ichimoku', 'Fibonacci', 'MACD', 'RSI',
            'red ribbon', 'blue ribbon', 'volume shelf', 'momentum bar',
            'Tenkin', 'GAN', 'whale momentum', 'retail momentum'
        ]
        
        for indicator in indicators:
            if indicator.lower() in text_content.lower():
                technical_data['indicators'].append(indicator)
        
        # Chart patterns
        patterns = [
            'breakout', 'consolidation', 'retrace', 'gap-up', 'expansion',
            'overextended', 'overbought', 'bullish', 'bearish'
        ]
        
        for pattern in patterns:
            if pattern.lower() in text_content.lower():
                technical_data['patterns'].append(pattern)
        
        # Timeframes
        timeframes = ['daily', 'weekly', 'monthly']
        for timeframe in timeframes:
            if timeframe.lower() in text_content.lower():
                technical_data['timeframes'].append(timeframe)
        
        return technical_data if any(technical_data.values()) else {}

class StructuredDataSchema:
    """Standardized data schema for consistent processing"""
    
    @staticmethod
    def create_stock_analysis_schema(symbol: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create standardized stock analysis structure"""
        return {
            "stock_analysis": {
                "symbol": symbol.upper(),
                "metadata": {
                    "channel_name": analysis_data.get('Channel name', analysis_data.get('channel_name', '')),
                    "date": analysis_data.get('Date', ''),
                    "frame_paths": analysis_data.get('frame_paths', []),
                    "time_range": {
                        "start_time": analysis_data.get('start_time'),
                        "end_time": analysis_data.get('end_time')
                    }
                },
                "price_data": {
                    "current_price": None,
                    "key_levels": {
                        "support": [],
                        "resistance": [],
                        "targets": []
                    }
                },
                "technical_analysis": {
                    "pattern": "",
                    "setup_quality": "",
                    "risk_assessment": "",
                    "targets": [],
                    "entry_triggers": []
                },
                "timeframe_analysis": {
                    "daily": "",
                    "weekly": "",
                    "monthly": ""
                },
                "original_summary": analysis_data.get('summary', ''),
                "original_key_points": analysis_data.get('key_points', [])
            }
        }
    
    @staticmethod
    def validate_schema(data: Dict[str, Any]) -> bool:
        """Validate data against expected schema"""
        required_fields = ['stock_analysis']
        
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field: {field}")
                return False
        
        stock_analysis = data['stock_analysis']
        required_stock_fields = ['symbol', 'metadata', 'technical_analysis']
        
        for field in required_stock_fields:
            if field not in stock_analysis:
                logger.warning(f"Missing required stock analysis field: {field}")
                return False
        
        return True

class DataValidator:
    """Validate data consistency between processing stages"""
    
    @staticmethod
    def validate_data_integrity(original: Dict[str, Any], processed: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that critical data hasn't been lost during processing"""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "missing_fields": []
        }
        
        # Check for missing critical fields
        critical_fields = DataPreservationMixin.CRITICAL_FIELDS
        for field in critical_fields:
            if field in original and field not in processed:
                validation_results["missing_fields"].append(field)
                validation_results["warnings"].append(f"Critical field '{field}' missing in processed data")
        
        # Check section count consistency
        orig_sections = original.get('sections', [])
        proc_sections = processed.get('sections', [])
        
        if len(orig_sections) != len(proc_sections):
            validation_results["warnings"].append(
                f"Section count mismatch: original={len(orig_sections)}, processed={len(proc_sections)}"
            )
        
        # Check stock symbol consistency
        orig_stocks = set()
        proc_stocks = set()
        
        for section in orig_sections:
            if 'stocks' in section:
                orig_stocks.update(section['stocks'])
        
        for section in proc_sections:
            if 'stocks' in section:
                proc_stocks.update(section['stocks'])
        
        missing_stocks = orig_stocks - proc_stocks
        if missing_stocks:
            validation_results["errors"].append(f"Missing stocks in processed data: {missing_stocks}")
            validation_results["valid"] = False
        
        return validation_results
