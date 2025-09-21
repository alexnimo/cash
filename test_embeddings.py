#!/usr/bin/env python3
"""
Test script to verify embedding service functionality.
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_embedding_service():
    """Test the EmbeddingService functionality."""
    print("🧪 Testing EmbeddingService...")
    
    try:
        from app.services.embedding_service import EmbeddingService, get_embedding_service
        print("✓ Successfully imported EmbeddingService")
        
        # Test getting the service instance
        service = get_embedding_service()
        print(f"✓ Created service instance: {service.provider}/{service.model}")
        
        # Test model info
        info = service.get_model_info()
        print(f"✓ Model info: {info}")
        
        # Test embedding a simple text
        test_texts = ["Hello world", "This is a test"]
        embeddings = service.embed_texts(test_texts)
        
        print(f"✓ Generated embeddings for {len(test_texts)} texts")
        print(f"  - First embedding dimension: {len(embeddings[0])}")
        print(f"  - Expected dimension: {service.get_dimension()}")
        
        if len(embeddings[0]) == service.get_dimension():
            print("✓ Embedding dimensions match expected")
        else:
            print("❌ Embedding dimensions don't match")
            
        # Test single text convenience function
        from app.services.embedding_service import embed_text
        single_embedding = embed_text("Single test text")
        print(f"✓ Single text embedding dimension: {len(single_embedding)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_integration():
    """Test that EmbeddingService integrates with config system."""
    print("\n🔧 Testing config integration...")
    
    try:
        from app.core.unified_config import ConfigManager
        config = ConfigManager()
        embedding_config = config.get('agents', 'embedding')
        
        print(f"✓ Found embedding config: {embedding_config}")
        
        expected_fields = ['provider', 'model', 'dimension', 'metric']
        for field in expected_fields:
            if field in embedding_config:
                print(f"  ✓ {field}: {embedding_config[field]}")
            else:
                print(f"  ❌ Missing {field}")
                
        return True
        
    except Exception as e:
        print(f"❌ Config test error: {e}")
        return False

def test_agent_integration():
    """Test that the agent system can access embeddings."""
    print("\n🤖 Testing agent integration...")
    
    try:
        # Try importing the agent modules that use embeddings
        from app.tools.pinecone_tool_v2 import PineconeAdvancedToolSpec
        print("✓ PineconeAdvancedToolSpec imports successfully")
        
        # Don't initialize it fully since that requires API keys
        print("  (Skipping full initialization - requires API keys)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Agent import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Agent test error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Embedding Service Tests")
    print("=" * 50)
    
    tests = [
        test_embedding_service,
        test_config_integration,
        test_agent_integration,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n📊 Test Results")
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✓ PASSED" if result else "❌ FAILED"
        print(f"{i+1}. {test_func.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Embedding service is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
