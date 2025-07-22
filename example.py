#!/usr/bin/env python3
"""
Simple Example Script for Document Similarity Detection
=====================================================

This script demonstrates basic functionality without the full main.py complexity.
Perfect for testing and understanding the system components.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_simple_example():
    """Run a simple example to test the system"""
    print("🚀 Document Similarity Detection - Simple Example")
    print("=" * 50)
    
    try:
        # Import modules
        from document_processor import DocumentProcessor
        from similarity_engine import SimilarityEngine
        from indexing import DocumentIndex
        
        print("✅ All modules imported successfully!")
        
        # Initialize components
        print("\n📚 Initializing components...")
        processor = DocumentProcessor()
        similarity_engine = SimilarityEngine()
        index = DocumentIndex()
        
        print("✅ Components initialized!")
        
        # Test with sample text documents
        print("\n📝 Testing with sample text documents...")
        
        sample_docs = {
            "doc1": "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "doc2": "Deep learning uses neural networks to process data and make predictions.",
            "doc3": "Natural language processing helps computers understand human language.",
            "doc4": "Machine learning algorithms can be used for classification and regression tasks.",
            "doc5": "The weather is nice today and I want to go for a walk in the park."
        }
        
        # Add documents to index
        for doc_id, content in sample_docs.items():
            success = index.add_document(doc_id, f"sample_{doc_id}.txt", content)
            if success:
                print(f"  ✅ Added {doc_id}")
            else:
                print(f"  ❌ Failed to add {doc_id}")
        
        # Build search index
        print("\n🔍 Building search index...")
        index.build_vector_index()
        print("✅ Search index built!")
        
        # Test search functionality
        print("\n🎯 Testing search functionality...")
        query = "machine learning algorithms"
        results = index.similarity_search(query, top_k=3)
        
        print(f"Query: '{query}'")
        print("Results:")
        for doc_id, similarity in results:
            print(f"  📄 {doc_id}: {similarity:.3f} similarity")
        
        # Test full-text search
        print(f"\n🔍 Testing full-text search...")
        text_results = index.search("neural networks", max_results=3)
        print("Full-text search results:")
        for doc_id, score in text_results:
            print(f"  📄 {doc_id}: {score:.3f} relevance score")
        
        # Show statistics
        print(f"\n📊 Index Statistics:")
        stats = index.get_index_statistics()
        print(f"  Total documents: {stats['total_documents']}")
        print(f"  Unique terms: {stats['unique_terms']}")
        print(f"  Memory usage: {stats['memory_usage_mb']:.1f} MB")
        
        print(f"\n✅ Simple example completed successfully!")
        print("You can now try running the full main.py with: python main.py demo")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_simple_example()
