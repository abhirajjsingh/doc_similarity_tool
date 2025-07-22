"""
Configuration settings for Document Similarity Detection System
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DOCS_DIR = DATA_DIR / "processed_docs"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CLUSTERS_DIR = DATA_DIR / "clusters"
SAMPLE_PDFS_DIR = DATA_DIR / "sample_pdfs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, PROCESSED_DOCS_DIR, EMBEDDINGS_DIR, CLUSTERS_DIR, SAMPLE_PDFS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Similarity thresholds
SIMILARITY_THRESHOLDS = {
    "high": 0.8,        # Very similar documents
    "medium": 0.6,      # Moderately similar
    "low": 0.4,         # Somewhat similar
    "default": 0.7      # Default threshold for clustering
}

# Text processing settings
TEXT_PROCESSING = {
    "min_doc_length": 100,           # Minimum characters in a document
    "max_doc_length": 1000000,       # Maximum characters to process
    "remove_stopwords": True,
    "min_df": 2,                     # Min document frequency for TF-IDF
    "max_df": 0.8,                   # Max document frequency for TF-IDF
    "max_features": 10000,           # Max features for TF-IDF
    "ngram_range": (1, 2),           # N-gram range for TF-IDF
}

# LSH settings for optimization
LSH_SETTINGS = {
    "num_perm": 128,                 # Number of permutations for MinHash
    "threshold": 0.7,                # LSH threshold
    "num_bands": 20,                 # Number of bands for LSH
    "band_width": 6,                 # Band width for LSH
}

# Clustering settings
CLUSTERING = {
    "algorithm": "threshold",         # "threshold", "hierarchical", "dbscan"
    "merge_threshold": 0.9,          # Threshold for merging clusters
    "min_cluster_size": 2,           # Minimum documents in a cluster
    "max_cluster_size": 50,          # Maximum documents in a cluster
}

# Performance settings
PERFORMANCE = {
    "batch_size": 100,               # Batch size for processing
    "n_jobs": -1,                    # Number of parallel jobs (-1 = all cores)
    "chunk_size": 1000,              # Chunk size for text processing
    "cache_embeddings": True,        # Cache TF-IDF vectors
    "use_sparse_matrix": True,       # Use sparse matrices for efficiency
}

# Alert system settings
ALERT_SYSTEM = {
    "check_new_docs": True,          # Check for new documents
    "similarity_threshold": 0.7,     # Threshold for alerts
    "max_similar_docs": 10,          # Max similar docs to return
    "include_snippet": True,         # Include text snippets in alerts
    "snippet_length": 200,           # Length of text snippets
}

# API settings
API_SETTINGS = {
    "host": "localhost",
    "port": 8000,
    "debug": True,
    "title": "Document Similarity API",
    "description": "API for document similarity detection and clustering",
    "version": "1.0.0",
}

# Logging settings
LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "similarity_system.log",
    "max_size": 10485760,            # 10MB
    "backup_count": 5,
}

# File extensions to process
SUPPORTED_EXTENSIONS = {
    ".txt", ".pdf", ".doc", ".docx", ".rtf"
}

# Similarity metrics to use
SIMILARITY_METRICS = [
    "cosine",           # Cosine similarity (primary)
    "jaccard",          # Jaccard similarity
    "euclidean",        # Euclidean distance
    "manhattan",        # Manhattan distance
]

# Feature extraction methods
FEATURE_METHODS = [
    "tfidf",            # TF-IDF (primary)
    "count",            # Count vectorizer
    "binary",           # Binary occurrence
    "ngram",            # N-gram features
]