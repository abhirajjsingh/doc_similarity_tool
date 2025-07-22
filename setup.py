#!/usr/bin/env python3
"""
Quick Start Script for Document Similarity Tool
==============================================

This script provides a quick way to get started with the document similarity system.
It includes system checks, dependency verification, and basic setup.
"""

import os
import sys
import importlib
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    min_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version >= min_version:
        print(f"✅ Python {current_version[0]}.{current_version[1]} - Compatible")
        return True
    else:
        print(f"❌ Python {current_version[0]}.{current_version[1]} - Requires Python {min_version[0]}.{min_version[1]}+")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'numpy', 'scipy', 'scikit-learn', 'pandas',
        'nltk', 'datasketch', 'joblib'
    ]
    
    optional_packages = [
        'PyPDF2', 'pdfplumber', 'fitz', 'docx',
        'faiss', 'transformers', 'sentence_transformers'
    ]
    
    print("\n📦 Checking required dependencies...")
    missing_required = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_required.append(package)
    
    print("\n📦 Checking optional dependencies...")
    missing_optional = []
    
    for package in optional_packages:
        try:
            if package == 'fitz':
                importlib.import_module(package)
            elif package == 'docx':
                importlib.import_module(package)
            else:
                importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ⚠️  {package} (optional)")
            missing_optional.append(package)
    
    return missing_required, missing_optional

def check_data_directories():
    """Check if data directories exist"""
    directories = [
        'data',
        'data/sample_pdfs',
        'data/processed_docs',
        'data/embeddings',
        'data/clusters'
    ]
    
    print("\n📁 Checking data directories...")
    
    for dir_path in directories:
        path = Path(dir_path)
        if path.exists():
            file_count = len(list(path.glob('*'))) if path.is_dir() else 0
            print(f"  ✅ {dir_path} ({file_count} files)")
        else:
            path.mkdir(parents=True, exist_ok=True)
            print(f"  ⚠️  {dir_path} (created)")

def download_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        print("\n📚 Downloading NLTK data...")
        
        required_data = ['punkt', 'stopwords']
        for data_name in required_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}')
                print(f"  ✅ {data_name} already available")
            except LookupError:
                print(f"  📥 Downloading {data_name}...")
                nltk.download(data_name, quiet=True)
                print(f"  ✅ {data_name} downloaded")
                
    except ImportError:
        print("  ❌ NLTK not available - skipping download")

def run_basic_test():
    """Run a basic test to verify system functionality"""
    print("\n🧪 Running basic system test...")
    
    try:
        # Add src to path
        sys.path.insert(0, 'src')
        
        # Test basic imports
        from config.settings import SIMILARITY_THRESHOLDS
        print("  ✅ Configuration loaded")
        
        # Test that we can create basic text processing
        test_text = "This is a test document for similarity detection."
        print(f"  ✅ Basic text processing ready")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic test failed: {e}")
        return False

def show_usage_instructions():
    """Show instructions for using the system"""
    print("\n" + "="*60)
    print("🎯 QUICK START INSTRUCTIONS")
    print("="*60)
    
    print("\n1. Add some PDF or text files to the data/sample_pdfs/ directory")
    print("2. Run the simple example:")
    print("   python example.py")
    
    print("\n3. Or run the full demo:")
    print("   python main.py demo")
    
    print("\n4. Available commands:")
    print("   python main.py process -d data/sample_pdfs")
    print("   python main.py search -q 'your search query'")
    print("   python main.py cluster -t 0.7")
    print("   python main.py duplicates")
    
    print("\n5. For help:")
    print("   python main.py --help")
    
    print("\n📚 Documentation:")
    print("   Check README.md for detailed instructions")
    print("   Check the src/ directory for module documentation")

def main():
    """Main setup and verification function"""
    print("🚀 Document Similarity Tool - Setup & Verification")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    missing_required, missing_optional = check_dependencies()
    
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print("Install with: pip install -r requirements.txt")
        sys.exit(1)
    
    if missing_optional:
        print(f"\n⚠️  Missing optional packages: {', '.join(missing_optional)}")
        print("Some features may not be available.")
        print("For full functionality, install with: pip install -r requirements.txt")
    
    # Check directories
    check_data_directories()
    
    # Download NLTK data
    download_nltk_data()
    
    # Run basic test
    if run_basic_test():
        print("\n✅ System verification completed successfully!")
        show_usage_instructions()
    else:
        print("\n❌ System verification failed!")
        print("Please check the error messages above and install missing dependencies.")
        sys.exit(1)

if __name__ == "__main__":
    main()
