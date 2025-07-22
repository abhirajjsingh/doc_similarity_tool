# Document Similarity Tool - Project Overview

## Created Files and Their Purpose

### Core Application Files

1. **main.py** - Main application entry point
   - Complete command-line interface with multiple modes
   - Orchestrates all system components
   - Provides demo, processing, search, clustering capabilities
   - Comprehensive error handling and logging

2. **example.py** - Simple example for quick testing
   - Basic functionality demonstration
   - Tests core components without complexity
   - Perfect for initial verification

3. **setup.py** - System setup and verification
   - Checks Python version compatibility
   - Verifies all dependencies
   - Downloads NLTK data
   - Creates necessary directories
   - Runs basic system tests

### User Interface Scripts

4. **run.bat** - Windows batch script for easy access
   - Interactive menu system
   - No need to remember command-line arguments
   - User-friendly interface for Windows users

5. **run.sh** - Unix/Linux/macOS shell script
   - Colored output for better user experience
   - Interactive menu system
   - Cross-platform compatibility

## How to Use the System

### Quick Start (Recommended)
1. First time setup:
   ```bash
   python setup.py
   ```

2. Run simple example:
   ```bash
   python example.py
   ```

3. Run full demo:
   ```bash
   python main.py demo
   ```

### Interactive Menu (Easiest)
- Windows: Double-click `run.bat`
- Unix/Linux/macOS: `./run.sh`

### Command Line Interface

#### Process documents:
```bash
python main.py process -d data/sample_pdfs
```

#### Search for similar documents:
```bash
python main.py search -q "machine learning" -k 5
```

#### Cluster documents:
```bash
python main.py cluster -t 0.7
```

#### Find duplicates:
```bash
python main.py duplicates -t 0.9
```

#### Export results:
```bash
python main.py export -o results/
```

## System Architecture

The main.py file orchestrates these components:

1. **DocumentProcessor** - Extracts text from PDFs, DOCs, etc.
2. **SimilarityEngine** - Computes document similarities using multiple metrics
3. **DocumentIndex** - Fast document storage and retrieval with search capabilities
4. **DocumentClustering** - Groups similar documents automatically
5. **AlertSystem** - Real-time similarity detection for new documents

## Key Features Implemented

### Document Processing
- Multi-format support (PDF, TXT, DOC, DOCX)
- Batch processing capabilities
- Error handling and recovery
- Processing statistics

### Similarity Detection
- Multiple similarity metrics (Cosine, Jaccard, etc.)
- LSH optimization for large datasets
- TF-IDF vectorization
- Optional FAISS integration for fast search

### Document Indexing
- Full-text search capabilities
- Vector-based similarity search
- Metadata management
- SQLite backend for persistence

### Clustering
- Threshold-based clustering
- Automatic cluster creation
- Cluster statistics and analysis

### Alert System
- Real-time similarity detection
- Configurable thresholds
- Multiple alert types and severity levels
- Export capabilities (JSON, CSV)

## Demonstration Workflow

When you run `python main.py demo`, the system:

1. **Processes sample documents** from `data/sample_pdfs/`
2. **Builds search indices** for fast retrieval
3. **Demonstrates similarity search** with example queries
4. **Performs document clustering** to group similar documents
5. **Shows alert system** functionality with new document simulation
6. **Displays comprehensive statistics** about the system performance

## Configuration

The system is highly configurable through `config/settings.py`:
- Similarity thresholds
- Text processing parameters
- LSH settings for optimization
- Clustering algorithms
- Performance tuning options

## Output Files

The system generates several output files:
- `similarity_system.log` - Comprehensive logging
- `data/alerts.json` - Alert system output
- `data/embeddings/` - Document vectors and indices
- `data/clusters/clustering_results_*.json` - Clustering results
- Export files in timestamped directories

## Error Handling

Comprehensive error handling includes:
- Dependency checking
- File availability verification
- Memory management
- Graceful fallbacks when optional components unavailable
- Detailed error logging

## Performance Features

- Batch processing for efficiency
- Caching to reduce computation
- Optional FAISS for fast similarity search
- Memory usage monitoring
- Processing time tracking

