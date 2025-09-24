#!/usr/bin/env python3
"""
Pre-load embedding models at startup to avoid runtime loading delays.
"""

import sys
import time
sys.path.append('/Users/akarna/Blackbox POC/src')

def preload_models():
    """Pre-load all models to avoid runtime delays."""
    
    print("🚀 Pre-loading Models for Fast Startup")
    print("=" * 50)
    
    try:
        # Import and initialize the embedding model manager
        from shared.services.embedding_model_manager import embedding_manager
        
        print("📥 Loading embedding model...")
        start_time = time.time()
        
        # This will load the model if not already loaded
        model = embedding_manager.model
        
        load_time = time.time() - start_time
        
        print(f"✅ Embedding model loaded in {load_time:.2f} seconds")
        
        # Get model info
        info = embedding_manager.get_model_info()
        print(f"📊 Model info:")
        print(f"   - Name: {info['model_name']}")
        print(f"   - Dimension: {info['dimension']}")
        print(f"   - Max sequence length: {info['max_seq_length']}")
        
        # Test encoding speed
        test_texts = [
            "Russian disinformation campaigns",
            "threat actors using social media",
            "Germany targeted by Doppelgänger"
        ]
        
        print(f"\n⚡ Testing encoding speed...")
        start_time = time.time()
        embeddings = embedding_manager.encode(test_texts)
        encode_time = time.time() - start_time
        
        print(f"✅ Encoded {len(test_texts)} texts in {encode_time:.3f} seconds")
        print(f"📈 Average time per text: {encode_time/len(test_texts):.3f} seconds")
        
        # Test single encoding
        start_time = time.time()
        single_embedding = embedding_manager.encode_single("Test query")
        single_time = time.time() - start_time
        
        print(f"🎯 Single text encoding: {single_time:.3f} seconds")
        
        print(f"\n{'='*50}")
        print("✅ Models pre-loaded successfully!")
        print("🚀 Application ready for fast vector search!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error pre-loading models: {e}")
        return False


if __name__ == "__main__":
    success = preload_models()
    if not success:
        sys.exit(1)
