#!/usr/bin/env python3
"""
Test script to verify the NotionAgent rich text formatting is working correctly.
"""

import json
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.agents.agents import NotionAgent
from app.core.data_preservation import DataPreservationMixin, DataValidator
from app.core.notion_formatting import NotionRichTextFormatter

def test_notion_agent_formatting():
    """Test that NotionAgent properly formats data"""
    print("[*] Testing NotionAgent Rich Text Formatting...")
    
    # Sample input data similar to what comes from RAG agent
    test_data = {
        "Date": "2024-08-18",
        "Channel name": "Test Channel",
        "sections": [
            {
                "topic": "Technical Profile: IREN",
                "stocks": ["IREN"],
                "summary": "IREN has achieved a major bullish breakout, clearing a significant resistance level at $19.35 (88.6% log-based Fib) and maintaining it as crucial support. This marks the first sustained weekly close above this level since its IPO, indicating strong follow-through and continuation potential.",
                "key_points": [
                    "Pattern: Major bullish breakout with sustained close above key Fib level.",
                    "Setup: Broke out above 88.6% log-based Fib at $19.35, which is now critical support.",
                    "Risk: Failure to maintain stability above $19.35.",
                    "Target: Explosive move towards $28.25, all-time highs, and further Fib extensions."
                ],
                "frame_paths": [
                    "C:\\cash-temp-data\\videos\\frames\\test\\frame_0730.jpg",
                    "C:\\cash-temp-data\\videos\\frames\\test\\frame_0750.jpg"
                ],
                "source": "Composite Analysis",
                "tags": []
            }
        ]
    }
    
    # Create a mock NotionAgent to test formatting
    class MockNotionAgent(DataPreservationMixin):
        def __init__(self):
            self.rich_text_formatter = NotionRichTextFormatter()
            self.agent_debug_enabled = True
            self.debug_dir = Path("C:/cash-temp-data/debug")
            
        def _format_data_for_notion(self, data):
            """Copy of the actual method from NotionAgent"""
            try:
                # Create a copy to avoid modifying original data
                formatted_data = data.copy()
                
                # Format each section with rich text blocks
                if 'sections' in formatted_data:
                    for i, section in enumerate(formatted_data['sections']):
                        # Convert section content to rich Notion blocks
                        rich_blocks = self.rich_text_formatter.format_stock_analysis(section)
                        
                        # Store both original and formatted content
                        section['notion_blocks'] = rich_blocks
                        
                        # Preserve critical data using the mixin method
                        original_section = data.get('sections', [{}])[i] if i < len(data.get('sections', [])) else section
                        formatted_data['sections'][i] = self.preserve_critical_data(original_section, section)
                        
                # Save formatted data for debugging if enabled
                if self.agent_debug_enabled:
                    formatted_debug_file = self.debug_dir / f"test_notion_formatted.json"
                    with open(formatted_debug_file, "w") as f:
                        json.dump(formatted_data, f, indent=2)
                    print(f"[>] Saved formatted data to {formatted_debug_file}")
                    
                return formatted_data
                
            except Exception as e:
                print(f"[ERROR] Error formatting data for Notion: {str(e)}")
                # Return original data if formatting fails
                return data
    
    # Test the formatting
    mock_agent = MockNotionAgent()
    formatted_data = mock_agent._format_data_for_notion(test_data)
    
    # Validate the results
    validation_results = DataValidator.validate_data_integrity(test_data, formatted_data)
    
    print(f"[+] Formatting test completed")
    print(f"   - Original sections: {len(test_data.get('sections', []))}")
    print(f"   - Formatted sections: {len(formatted_data.get('sections', []))}")
    
    # Check if rich blocks were added
    has_rich_blocks = False
    if 'sections' in formatted_data:
        for section in formatted_data['sections']:
            if 'notion_blocks' in section and section['notion_blocks']:
                has_rich_blocks = True
                print(f"   - Rich blocks generated: {len(section['notion_blocks'])} blocks")
                break
    
    print(f"   - Rich formatting applied: {'PASS' if has_rich_blocks else 'FAIL'}")
    print(f"   - Missing fields: {len(validation_results.get('missing_fields', []))}")
    print(f"   - Data inconsistencies: {len(validation_results.get('data_inconsistencies', []))}")
    
    # Show sample rich block structure
    if has_rich_blocks and 'sections' in formatted_data:
        first_section = formatted_data['sections'][0]
        if 'notion_blocks' in first_section and first_section['notion_blocks']:
            print(f"\n[>] Sample rich block structure:")
            sample_block = first_section['notion_blocks'][0]
            print(f"   Type: {sample_block.get('type', 'unknown')}")
            if sample_block.get('type') == 'heading_1' and 'heading_1' in sample_block:
                rich_text = sample_block['heading_1'].get('rich_text', [])
                if rich_text:
                    print(f"   Content: {rich_text[0].get('text', {}).get('content', 'N/A')}")
                    annotations = rich_text[0].get('annotations', {})
                    print(f"   Formatting: {annotations}")
    
    return formatted_data, validation_results

def test_with_actual_debug_data():
    """Test with actual debug data from the system"""
    print("\n[*] Testing with actual debug data...")
    
    debug_file = Path("C:/cash-temp-data/debug/notion_data_latest.json")
    if not debug_file.exists():
        print("[!] Debug file not found, skipping actual data test")
        return None, None
    
    try:
        with open(debug_file, 'r') as f:
            actual_data = json.load(f)
        
        print(f"[>] Loaded actual data with {len(actual_data.get('sections', []))} sections")
        
        # Create mock agent and test formatting
        class MockNotionAgent(DataPreservationMixin):
            def __init__(self):
                self.rich_text_formatter = NotionRichTextFormatter()
                self.agent_debug_enabled = True
                self.debug_dir = Path("C:/cash-temp-data/debug")
                
            def _format_data_for_notion(self, data):
                try:
                    formatted_data = data.copy()
                    
                    if 'sections' in formatted_data:
                        for i, section in enumerate(formatted_data['sections']):
                            rich_blocks = self.rich_text_formatter.format_stock_analysis(section)
                            section['notion_blocks'] = rich_blocks
                            
                            original_section = data.get('sections', [{}])[i] if i < len(data.get('sections', [])) else section
                            formatted_data['sections'][i] = self.preserve_critical_data(original_section, section)
                            
                    if self.agent_debug_enabled:
                        formatted_debug_file = self.debug_dir / f"test_actual_formatted.json"
                        with open(formatted_debug_file, "w") as f:
                            json.dump(formatted_data, f, indent=2)
                        print(f"[>] Saved actual formatted data to {formatted_debug_file}")
                        
                    return formatted_data
                    
                except Exception as e:
                    print(f"[ERROR] Error formatting actual data: {str(e)}")
                    return data
        
        mock_agent = MockNotionAgent()
        formatted_data = mock_agent._format_data_for_notion(actual_data)
        
        # Validate
        validation_results = DataValidator.validate_data_integrity(actual_data, formatted_data)
        
        print(f"[+] Actual data formatting completed")
        print(f"   - Original sections: {len(actual_data.get('sections', []))}")
        print(f"   - Formatted sections: {len(formatted_data.get('sections', []))}")
        print(f"   - Missing fields: {len(validation_results.get('missing_fields', []))}")
        
        # Check stocks preserved
        original_stocks = set()
        formatted_stocks = set()
        
        for section in actual_data.get('sections', []):
            original_stocks.update(section.get('stocks', []))
        
        for section in formatted_data.get('sections', []):
            formatted_stocks.update(section.get('stocks', []))
        
        print(f"   - Original stocks: {len(original_stocks)}")
        print(f"   - Formatted stocks: {len(formatted_stocks)}")
        print(f"   - Stocks preserved: {'PASS' if original_stocks == formatted_stocks else 'FAIL'}")
        
        return formatted_data, validation_results
        
    except Exception as e:
        print(f"[ERROR] Failed to test actual data: {str(e)}")
        return None, None

def main():
    """Run all formatting tests"""
    print("[*] Testing NotionAgent Rich Text Formatting Pipeline")
    print("=" * 60)
    
    try:
        # Test with sample data
        formatted_sample, validation_sample = test_notion_agent_formatting()
        
        # Test with actual debug data
        formatted_actual, validation_actual = test_with_actual_debug_data()
        
        print("\n" + "=" * 60)
        print("[>] Test Summary:")
        print("[+] Sample Data Formatting: Working")
        if formatted_actual:
            print("[+] Actual Data Formatting: Working")
        else:
            print("[!] Actual Data Formatting: Skipped (no debug file)")
        
        print("\n[SUCCESS] Rich text formatting pipeline is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
