#!/usr/bin/env python3
"""
Script to download and cache the embedding model locally.
This prevents downloading the model at runtime.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append('/Users/akarna/Blackbox POC/src')

def download_embedding_model():
    """Download and cache the embedding model locally."""
    
    print("🔄 Downloading embedding model locally...")
    print("=" * 50)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Model name used in the application
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        
        print(f"📥 Downloading model: {model_name}")
        print("This may take a few minutes on first run...")
        
        # Download and cache the model
        # This will automatically cache it in ~/.cache/huggingface/transformers/
        model = SentenceTransformer(model_name)
        
        # Test the model to ensure it's working
        test_text = "This is a test sentence for embedding."
        embedding = model.encode(test_text)
        
        print(f"✅ Model downloaded and cached successfully!")
        print(f"📊 Embedding dimension: {model.get_sentence_embedding_dimension()}")
        print(f"🧪 Test embedding shape: {embedding.shape}")
        
        # Show cache location
        cache_path = Path.home() / ".cache" / "huggingface" / "transformers"
        print(f"📁 Cache directory: {cache_path}")
        print(f"💾 Model cached locally for fast loading")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False


def verify_model_cache():
    """Verify that the model is cached and can be loaded quickly."""
    
    print("\n🔍 Verifying model cache...")
    print("-" * 30)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        
        # Time the loading
        import time
        start_time = time.time()
        
        model = SentenceTransformer(model_name)
        
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        print(f"📊 Model info:")
        print(f"   - Dimension: {model.get_sentence_embedding_dimension()}")
        print(f"   - Max sequence length: {model.max_seq_length}")
        
        # Test encoding
        test_texts = [
            "Russian disinformation campaigns",
            "threat actors using social media",
            "Germany targeted by Doppelgänger"
        ]
        
        start_time = time.time()
        embeddings = model.encode(test_texts)
        encode_time = time.time() - start_time
        
        print(f"⚡ Encoded {len(test_texts)} texts in {encode_time:.3f} seconds")
        print(f"📈 Average time per text: {encode_time/len(test_texts):.3f} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying model: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Embedding Model Setup")
    print("=" * 50)
    
    # Download the model
    success = download_embedding_model()
    
    if success:
        # Verify it's working
        verify_model_cache()
        
        print(f"\n{'='*50}")
        print("✅ Setup Complete!")
        print("\n📋 Next steps:")
        print("1. The model is now cached locally")
        print("2. Future runs will load from cache (much faster)")
        print("3. No more runtime downloads!")
        print("4. You can now start your application")
    else:
        print(f"\n{'='*50}")
        print("❌ Setup Failed!")
        print("Please check your internet connection and try again.")
