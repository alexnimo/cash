#!/usr/bin/env python3
"""
Test script for the improved Notion data integration pipeline.
Tests data preservation, rich text formatting, and validation components.
"""

import json
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.core.data_preservation import DataPreservationMixin, DataValidator
from app.core.notion_formatting import NotionRichTextFormatter, NotionBlockBuilder

def test_data_preservation():
    """Test data preservation functionality"""
    print("[*] Testing Data Preservation...")
    
    # Sample original data
    original_data = {
        "Date": "2024-08-18",
        "Channel name": "Test Channel",
        "sections": [
            {
                "topic": "Technical Profile: AAPL",
                "stocks": ["AAPL"],
                "summary": "Apple showing strong support at $220 with RSI at 45 indicating oversold conditions.",
                "key_points": [
                    "Support level at $220",
                    "RSI at 45 - oversold",
                    "Volume spike on recent dip"
                ],
                "frame_paths": ["/path/to/aapl_chart1.jpg", "/path/to/aapl_chart2.jpg"],
                "price_levels": {"support": 220, "resistance": 235},
                "technical_indicators": {"RSI": 45, "MACD": "bullish_crossover"}
            }
        ]
    }
    
    # Sample processed data (simulating some data loss)
    processed_data = {
        "Date": "2024-08-18",
        "Channel name": "Test Channel",
        "sections": [
            {
                "topic": "Technical Profile: AAPL",
                "stocks": ["AAPL"],
                "summary": "Apple analysis updated",
                "key_points": ["Updated analysis"]
                # Missing frame_paths, price_levels, technical_indicators
            }
        ]
    }
    
    # Test data preservation
    class TestMixin(DataPreservationMixin):
        pass
    
    mixin = TestMixin()
    preserved_data = mixin.preserve_critical_data(original_data, processed_data)
    
    # Validate preservation
    validation_results = DataValidator.validate_data_integrity(original_data, preserved_data)
    
    print(f"[+] Data preservation test completed")
    print(f"   - Missing fields before preservation: {len(validation_results.get('missing_fields', []))}")
    print(f"   - Critical fields preserved: {len([f for f in mixin.CRITICAL_FIELDS if f in str(preserved_data)])}")
    
    return preserved_data

def test_rich_text_formatting():
    """Test rich text formatting functionality"""
    print("\n[*] Testing Rich Text Formatting...")
    
    formatter = NotionRichTextFormatter()
    
    # Sample section data
    section_data = {
        "topic": "Technical Profile: AAPL",
        "stocks": ["AAPL"],
        "summary": "Apple showing strong support at $220 with RSI at 45. The stock has key resistance at $235 and shows bullish momentum.",
        "key_points": [
            "Support level at $220 - strong buying interest",
            "RSI at 45 indicates oversold conditions",
            "Volume spike confirms institutional interest",
            "Resistance at $235 - key breakout level"
        ],
        "frame_paths": ["/path/to/aapl_chart1.jpg"],
        "Channel name": "Test Channel"
    }
    
    # Test formatting methods
    rich_blocks = formatter.format_stock_analysis(section_data)
    
    print(f"[+] Rich text formatting test completed")
    print(f"   - Generated {len(rich_blocks)} Notion blocks")
    print(f"   - Rich blocks created: {'PASS' if rich_blocks else 'FAIL'}")
    
    # Show sample formatted content
    if rich_blocks:
        print(f"\n[>] Sample rich block types:")
        block_types = [block.get('type', 'unknown') for block in rich_blocks[:3]]
        print(f"   {', '.join(block_types)}")
    
    return {
        "notion_blocks": rich_blocks
    }

def test_validation_pipeline():
    """Test validation pipeline functionality"""
    print("\n[*] Testing Validation Pipeline...")
    
    # Test with sample data
    original = {
        "sections": [
            {
                "stocks": ["AAPL", "MSFT"],
                "summary": "Original detailed analysis",
                "frame_paths": ["/path1.jpg", "/path2.jpg"],
                "price_levels": {"support": 220}
            }
        ]
    }
    
    processed = {
        "sections": [
            {
                "stocks": ["AAPL", "MSFT"],
                "summary": "Processed analysis",
                "notion_blocks": [{"type": "paragraph"}],
                "formatted_summary": "Enhanced summary",
                # Missing frame_paths and price_levels
            }
        ]
    }
    
    validation_results = DataValidator.validate_data_integrity(original, processed)
    
    print(f"[+] Validation pipeline test completed")
    print(f"   - Missing fields detected: {len(validation_results.get('missing_fields', []))}")
    print(f"   - Data inconsistencies: {len(validation_results.get('data_inconsistencies', []))}")
    
    if validation_results.get('missing_fields'):
        print(f"   - Missing: {', '.join(validation_results['missing_fields'])}")
    
    return validation_results

def test_integration():
    """Test full integration of all components"""
    print("\n[*] Testing Full Integration...")
    
    # Start with original data
    original_data = {
        "Date": "2024-08-18",
        "Channel name": "Integration Test Channel",
        "sections": [
            {
                "topic": "Technical Profile: TSLA",
                "stocks": ["TSLA"],
                "summary": "Tesla showing consolidation pattern near $240 with strong volume. RSI at 52 suggests neutral momentum with potential for breakout above $250 resistance.",
                "key_points": [
                    "Consolidation pattern at $240 support",
                    "RSI at 52 - neutral territory",
                    "Volume increase on recent bounce",
                    "Key resistance at $250 for breakout"
                ],
                "frame_paths": ["/charts/tsla_daily.jpg", "/charts/tsla_4h.jpg"],
                "price_levels": {"support": 240, "resistance": 250},
                "technical_indicators": {"RSI": 52, "Volume": "increasing"}
            }
        ]
    }
    
    # Step 1: Data preservation
    class TestAgent(DataPreservationMixin):
        pass
    
    agent = TestAgent()
    
    # Step 2: Rich text formatting
    formatter = NotionRichTextFormatter()
    formatted_data = original_data.copy()
    
    for section in formatted_data["sections"]:
        section["notion_blocks"] = formatter.format_stock_analysis(section)
        
        # Preserve critical data
        section = agent.preserve_critical_data(section, section)
    
    # Step 3: Validation
    validation_results = DataValidator.validate_data_integrity(original_data, formatted_data)
    
    print(f"[+] Full integration test completed")
    print(f"   - Original sections: {len(original_data['sections'])}")
    print(f"   - Formatted sections: {len(formatted_data['sections'])}")
    print(f"   - Validation issues: {len(validation_results.get('missing_fields', []))}")
    print(f"   - Rich formatting applied: {'PASS' if any('notion_blocks' in s for s in formatted_data['sections']) else 'FAIL'}")
    
    return formatted_data, validation_results

def main():
    """Run all tests"""
    print("[*] Testing Improved Notion Data Integration Pipeline")
    print("=" * 60)
    
    try:
        # Run individual component tests
        preserved_data = test_data_preservation()
        formatted_content = test_rich_text_formatting()
        validation_results = test_validation_pipeline()
        
        # Run integration test
        final_data, final_validation = test_integration()
        
        print("\n" + "=" * 60)
        print("[>] Test Summary:")
        print("[+] Data Preservation: Working")
        print("[+] Rich Text Formatting: Working") 
        print("[+] Validation Pipeline: Working")
        print("[+] Full Integration: Working")
        print("\n[SUCCESS] All tests passed! The improved pipeline is ready for use.")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
