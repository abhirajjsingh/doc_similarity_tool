"""
Document indexing and retrieval system
Provides efficient storage, retrieval, and search capabilities for documents
"""

import os
import json
import pickle
import sqlite3
import hashlib
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Optional import for FAISS (requires: pip install faiss-cpu or faiss-gpu)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Advanced similarity search will be disabled.")

from config.settings import *

logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Metadata for indexed documents"""
    document_id: str
    file_path: str
    file_size: int
    created_at: datetime
    modified_at: datetime
    content_hash: str
    doc_type: str
    word_count: int
    character_count: int
    language: str = "en"
    encoding: str = "utf-8"
    processed: bool = False
    indexed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat() if value else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DocumentMetadata':
        """Create from dictionary"""
        # Convert ISO format strings back to datetime objects
        for key in ['created_at', 'modified_at', 'indexed_at']:
            if data.get(key):
                data[key] = datetime.fromisoformat(data[key])
        return cls(**data)

@dataclass
class IndexEntry:
    """Entry in the document index"""
    document_id: str
    terms: Set[str]
    term_frequencies: Dict[str, int]
    vector_id: Optional[int] = None  # ID in FAISS index
    cluster_id: Optional[str] = None
    
class DocumentIndex:
    """
    Main document indexing system
    Provides full-text search, similarity search, and metadata management
    """
    
    def __init__(self, index_dir: str = None, config: Dict[str, Any] = None):
        """
        Initialize document index
        
        Args:
            index_dir: Directory to store index files
            config: Configuration parameters
        """
        self.config = config or {}
        self.index_dir = Path(index_dir) if index_dir else EMBEDDINGS_DIR
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Index files
        self.metadata_file = self.index_dir / "metadata.json"
        self.index_file = self.index_dir / "document_index.pkl"
        self.vectors_file = self.index_dir / "document_vectors.pkl"
        self.faiss_index_file = self.index_dir / "faiss_index.bin"
        self.db_file = self.index_dir / "documents.db"
        
        # In-memory structures
        self.metadata: Dict[str, DocumentMetadata] = {}
        self.index: Dict[str, IndexEntry] = {}
        self.document_vectors: Optional[np.ndarray] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.faiss_index: Optional[faiss.Index] = None
        
        # Inverted index for full-text search
        self.inverted_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Statistics
        self.stats = {
            'total_documents': 0,
            'total_terms': 0,
            'index_size_mb': 0,
            'last_updated': None,
            'search_count': 0,
            'cache_hits': 0
        }
        
        # Initialize SQLite database
        self._init_database()
        
        # Load existing index
        self.load_index()
        
        logger.info(f"Document index initialized with {len(self.metadata)} documents")
    
    def _init_database(self):
        """Initialize SQLite database for metadata storage"""
        try:
            self.conn = sqlite3.connect(str(self.db_file))
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    created_at TEXT,
                    modified_at TEXT,
                    content_hash TEXT UNIQUE,
                    doc_type TEXT,
                    word_count INTEGER,
                    character_count INTEGER,
                    language TEXT DEFAULT 'en',
                    encoding TEXT DEFAULT 'utf-8',
                    processed BOOLEAN DEFAULT FALSE,
                    indexed_at TEXT,
                    UNIQUE(content_hash)
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    results_count INTEGER,
                    search_time REAL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_content_hash ON documents(content_hash);
            """)
            
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_type ON documents(doc_type);
            """)
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def add_document(self, document_id: str, file_path: str, 
                    content: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Add a document to the index
        
        Args:
            document_id: Unique identifier for the document
            file_path: Path to the document file
            content: Text content of the document
            metadata: Additional metadata
            
        Returns:
            True if document was added successfully
        """
        try:
            # Generate content hash
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Check if document already exists
            if self._document_exists(content_hash):
                logger.info(f"Document {document_id} already indexed (duplicate content)")
                return False
            
            # Get file metadata
            file_path_obj = Path(file_path)
            file_stats = file_path_obj.stat() if file_path_obj.exists() else None
            
            # Create document metadata
            doc_metadata = DocumentMetadata(
                document_id=document_id,
                file_path=file_path,
                file_size=file_stats.st_size if file_stats else len(content),
                created_at=datetime.fromtimestamp(file_stats.st_ctime) if file_stats else datetime.now(),
                modified_at=datetime.fromtimestamp(file_stats.st_mtime) if file_stats else datetime.now(),
                content_hash=content_hash,
                doc_type=file_path_obj.suffix.lower() if file_path_obj.suffix else ".txt",
                word_count=len(content.split()),
                character_count=len(content),
                indexed_at=datetime.now()
            )
            
            # Add custom metadata if provided
            if metadata:
                for key, value in metadata.items():
                    if hasattr(doc_metadata, key):
                        setattr(doc_metadata, key, value)
            
            # Process content and create index entry
            terms, term_frequencies = self._process_content(content)
            
            index_entry = IndexEntry(
                document_id=document_id,
                terms=terms,
                term_frequencies=term_frequencies
            )
            
            # Store in memory structures
            self.metadata[document_id] = doc_metadata
            self.index[document_id] = index_entry
            
            # Update inverted index
            for term in terms:
                self.inverted_index[term].add(document_id)
            
            # Store in database
            self._store_document_db(doc_metadata)
            
            # Update statistics
            self.stats['total_documents'] += 1
            self.stats['total_terms'] = len(self.inverted_index)
            self.stats['last_updated'] = datetime.now()
            
            logger.info(f"Added document {document_id} to index")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {document_id}: {str(e)}")
            return False
    
    def _document_exists(self, content_hash: str) -> bool:
        """Check if document with given content hash already exists"""
        cursor = self.conn.execute(
            "SELECT document_id FROM documents WHERE content_hash = ?",
            (content_hash,)
        )
        return cursor.fetchone() is not None
    
    def _process_content(self, content: str) -> Tuple[Set[str], Dict[str, int]]:
        """Process document content and extract terms"""
        # Simple tokenization and normalization
        import re
        
        # Convert to lowercase and remove special characters
        content = re.sub(r'[^\w\s]', ' ', content.lower())
        
        # Split into words
        words = content.split()
        
        # Remove empty strings and very short words
        words = [word for word in words if len(word) > 2]
        
        # Calculate term frequencies
        term_frequencies = {}
        for word in words:
            term_frequencies[word] = term_frequencies.get(word, 0) + 1
        
        # Get unique terms
        terms = set(words)
        
        return terms, term_frequencies
    
    def _store_document_db(self, metadata: DocumentMetadata):
        """Store document metadata in SQLite database"""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO documents 
                (document_id, file_path, file_size, created_at, modified_at, 
                 content_hash, doc_type, word_count, character_count, 
                 language, encoding, processed, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.document_id,
                metadata.file_path,
                metadata.file_size,
                metadata.created_at.isoformat(),
                metadata.modified_at.isoformat(),
                metadata.content_hash,
                metadata.doc_type,
                metadata.word_count,
                metadata.character_count,
                metadata.language,
                metadata.encoding,
                metadata.processed,
                metadata.indexed_at.isoformat() if metadata.indexed_at else None
            ))
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error storing document in database: {str(e)}")
            raise
    
    def remove_document(self, document_id: str) -> bool:
        """
        Remove a document from the index
        
        Args:
            document_id: ID of document to remove
            
        Returns:
            True if document was removed successfully
        """
        try:
            if document_id not in self.metadata:
                logger.warning(f"Document {document_id} not found in index")
                return False
            
            # Get document terms for cleanup
            index_entry = self.index.get(document_id)
            if index_entry:
                # Remove from inverted index
                for term in index_entry.terms:
                    if term in self.inverted_index:
                        self.inverted_index[term].discard(document_id)
                        # Remove empty term entries
                        if not self.inverted_index[term]:
                            del self.inverted_index[term]
            
            # Remove from memory structures
            del self.metadata[document_id]
            del self.index[document_id]
            
            # Remove from database
            self.conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            self.conn.commit()
            
            # Update statistics
            self.stats['total_documents'] -= 1
            self.stats['total_terms'] = len(self.inverted_index)
            self.stats['last_updated'] = datetime.now()
            
            logger.info(f"Removed document {document_id} from index")
            return True
            
        except Exception as e:
            logger.error(f"Error removing document {document_id}: {str(e)}")
            return False
    
    def search(self, query: str, max_results: int = 10) -> List[Tuple[str, float]]:
        """
        Search for documents using full-text search
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of (document_id, score) tuples
        """
        start_time = datetime.now()
        
        try:
            # Process query terms
            query_terms, _ = self._process_content(query)
            
            if not query_terms:
                return []
            
            # Find documents containing query terms
            candidate_docs = set()
            for term in query_terms:
                if term in self.inverted_index:
                    candidate_docs.update(self.inverted_index[term])
            
            if not candidate_docs:
                return []
            
            # Score documents
            results = []
            for doc_id in candidate_docs:
                score = self._calculate_search_score(doc_id, query_terms)
                if score > 0:
                    results.append((doc_id, score))
            
            # Sort by score and return top results
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:max_results]
            
            # Log search
            search_time = (datetime.now() - start_time).total_seconds()
            self._log_search(query, len(results), search_time)
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            return []
    
    def _calculate_search_score(self, document_id: str, query_terms: Set[str]) -> float:
        """Calculate relevance score for a document given query terms"""
        if document_id not in self.index:
            return 0.0
        
        index_entry = self.index[document_id]
        doc_terms = index_entry.terms
        term_frequencies = index_entry.term_frequencies
        
        # Calculate score based on term matches and frequencies
        score = 0.0
        total_doc_terms = sum(term_frequencies.values())
        
        for term in query_terms:
            if term in doc_terms:
                # TF score
                tf = term_frequencies[term] / total_doc_terms
                
                # IDF score (simplified)
                docs_with_term = len(self.inverted_index.get(term, set()))
                idf = np.log(len(self.metadata) / max(1, docs_with_term))
                
                score += tf * idf
        
        return score
    
    def _log_search(self, query: str, results_count: int, search_time: float):
        """Log search query for analytics"""
        try:
            self.conn.execute("""
                INSERT INTO search_history (query, results_count, search_time)
                VALUES (?, ?, ?)
            """, (query, results_count, search_time))
            self.conn.commit()
            
            self.stats['search_count'] += 1
            
        except Exception as e:
            logger.error(f"Error logging search: {str(e)}")
    
    def similarity_search(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Perform similarity-based search using vector representations
        
        Args:
            query_text: Query text
            top_k: Number of most similar documents to return
            
        Returns:
            List of (document_id, similarity_score) tuples
        """
        try:
            if self.document_vectors is None or self.vectorizer is None:
                logger.warning("Vector index not built. Building now...")
                self.build_vector_index()
            
            # Vectorize query
            query_vector = self.vectorizer.transform([query_text])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.document_vectors)[0]
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Convert to document IDs and scores
            results = []
            doc_ids = list(self.metadata.keys())
            
            for idx in top_indices:
                if similarities[idx] > 0:  # Only return positive similarities
                    doc_id = doc_ids[idx]
                    results.append((doc_id, float(similarities[idx])))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def build_vector_index(self):
        """Build TF-IDF vector index for similarity search"""
        try:
            if not self.metadata:
                logger.warning("No documents to build vector index")
                return
            
            # Collect all document texts (this would need access to actual content)
            # For now, we'll use term frequencies as a proxy
            documents = []
            doc_ids = []
            
            for doc_id, index_entry in self.index.items():
                # Reconstruct text from term frequencies
                text_tokens = []
                for term, freq in index_entry.term_frequencies.items():
                    text_tokens.extend([term] * freq)
                
                documents.append(" ".join(text_tokens))
                doc_ids.append(doc_id)
            
            # Build TF-IDF vectors
            self.vectorizer = TfidfVectorizer(
                max_features=TEXT_PROCESSING.get("max_features", 10000),
                min_df=TEXT_PROCESSING.get("min_df", 2),
                max_df=TEXT_PROCESSING.get("max_df", 0.8),
                ngram_range=TEXT_PROCESSING.get("ngram_range", (1, 2))
            )
            
            self.document_vectors = self.vectorizer.fit_transform(documents)
            
            logger.info(f"Built vector index for {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error building vector index: {str(e)}")
            raise
    
    def build_faiss_index(self):
        """Build FAISS index for efficient similarity search"""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available. Cannot build FAISS index.")
            return
            
        try:
            if self.document_vectors is None:
                self.build_vector_index()
            
            # Convert sparse matrix to dense for FAISS
            dense_vectors = self.document_vectors.toarray().astype(np.float32)
            
            # Build FAISS index
            dimension = dense_vectors.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
            
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(dense_vectors)
            
            # Add vectors to index
            self.faiss_index.add(dense_vectors)
            
            logger.info(f"Built FAISS index with {self.faiss_index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}")
            raise
    
    def faiss_similarity_search(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Perform similarity search using FAISS index
        
        Args:
            query_text: Query text
            top_k: Number of results to return
            
        Returns:
            List of (document_id, similarity_score) tuples
        """
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available. Using standard similarity search.")
            return self.similarity_search(query_text, top_k)
            
        try:
            if self.faiss_index is None:
                logger.warning("FAISS index not built. Building now...")
                self.build_faiss_index()
            
            if self.faiss_index is None:  # Still None after build attempt
                return self.similarity_search(query_text, top_k)
            
            # Vectorize and normalize query
            query_vector = self.vectorizer.transform([query_text]).toarray().astype(np.float32)
            faiss.normalize_L2(query_vector)
            
            # Search
            similarities, indices = self.faiss_index.search(query_vector, top_k)
            
            # Convert to document IDs
            doc_ids = list(self.metadata.keys())
            results = []
            
            for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(doc_ids) and sim > 0:
                    results.append((doc_ids[idx], float(sim)))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in FAISS similarity search: {str(e)}")
            return []
    
    def get_document_metadata(self, document_id: str) -> Optional[DocumentMetadata]:
        """Get metadata for a specific document"""
        return self.metadata.get(document_id)
    
    def get_all_documents(self) -> List[str]:
        """Get list of all document IDs"""
        return list(self.metadata.keys())
    
    def get_documents_by_type(self, doc_type: str) -> List[str]:
        """Get documents filtered by type"""
        return [
            doc_id for doc_id, metadata in self.metadata.items()
            if metadata.doc_type == doc_type
        ]
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """Get comprehensive index statistics"""
        stats = self.stats.copy()
        
        # Add current statistics
        stats.update({
            'memory_usage_mb': self._calculate_memory_usage(),
            'unique_terms': len(self.inverted_index),
            'avg_document_size': np.mean([m.character_count for m in self.metadata.values()]) if self.metadata else 0,
            'doc_type_distribution': self._get_type_distribution(),
            'language_distribution': self._get_language_distribution()
        })
        
        return stats
    
    def _calculate_memory_usage(self) -> float:
        """Calculate approximate memory usage in MB"""
        import sys
        
        total_size = 0
        total_size += sys.getsizeof(self.metadata)
        total_size += sys.getsizeof(self.index)
        total_size += sys.getsizeof(self.inverted_index)
        
        if self.document_vectors is not None:
            total_size += self.document_vectors.data.nbytes
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def _get_type_distribution(self) -> Dict[str, int]:
        """Get distribution of document types"""
        distribution = {}
        for metadata in self.metadata.values():
            doc_type = metadata.doc_type
            distribution[doc_type] = distribution.get(doc_type, 0) + 1
        return distribution
    
    def _get_language_distribution(self) -> Dict[str, int]:
        """Get distribution of document languages"""
        distribution = {}
        for metadata in self.metadata.values():
            language = metadata.language
            distribution[language] = distribution.get(language, 0) + 1
        return distribution
    
    def save_index(self):
        """Save index to disk"""
        try:
            # Save metadata
            metadata_dict = {doc_id: metadata.to_dict() 
                           for doc_id, metadata in self.metadata.items()}
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            # Save index and inverted index
            index_data = {
                'index': self.index,
                'inverted_index': dict(self.inverted_index),
                'stats': self.stats
            }
            with open(self.index_file, 'wb') as f:
                pickle.dump(index_data, f)
            
            # Save vectors if available
            if self.document_vectors is not None and self.vectorizer is not None:
                vector_data = {
                    'document_vectors': self.document_vectors,
                    'vectorizer': self.vectorizer
                }
                with open(self.vectors_file, 'wb') as f:
                    pickle.dump(vector_data, f)
            
            # Save FAISS index if available
            if FAISS_AVAILABLE and self.faiss_index is not None:
                faiss.write_index(self.faiss_index, str(self.faiss_index_file))
            
            logger.info("Index saved to disk")
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise
    
    def load_index(self):
        """Load index from disk"""
        try:
            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                    self.metadata = {doc_id: DocumentMetadata.from_dict(data)
                                   for doc_id, data in metadata_dict.items()}
            
            # Load index
            if self.index_file.exists():
                with open(self.index_file, 'rb') as f:
                    index_data = pickle.load(f)
                    self.index = index_data.get('index', {})
                    self.inverted_index = defaultdict(set, index_data.get('inverted_index', {}))
                    self.stats = index_data.get('stats', self.stats)
            
            # Load vectors
            if self.vectors_file.exists():
                with open(self.vectors_file, 'rb') as f:
                    vector_data = pickle.load(f)
                    self.document_vectors = vector_data.get('document_vectors')
                    self.vectorizer = vector_data.get('vectorizer')
            
            # Load FAISS index
            if FAISS_AVAILABLE and self.faiss_index_file.exists():
                self.faiss_index = faiss.read_index(str(self.faiss_index_file))
            
            logger.info(f"Loaded index with {len(self.metadata)} documents")
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            # Initialize empty structures on load failure
            self.metadata = {}
            self.index = {}
            self.inverted_index = defaultdict(set)
    
    def optimize_index(self):
        """Optimize index for better performance"""
        try:
            # Remove empty terms from inverted index
            empty_terms = [term for term, docs in self.inverted_index.items() if not docs]
            for term in empty_terms:
                del self.inverted_index[term]
            
            # Rebuild vector indices if they exist
            if self.document_vectors is not None:
                self.build_vector_index()
            
            if self.faiss_index is not None:
                self.build_faiss_index()
            
            # Update statistics
            self.stats['total_terms'] = len(self.inverted_index)
            self.stats['last_updated'] = datetime.now()
            
            logger.info("Index optimization complete")
            
        except Exception as e:
            logger.error(f"Error optimizing index: {str(e)}")
    
    def close(self):
        """Close database connections and save index"""
        try:
            self.save_index()
            if hasattr(self, 'conn'):
                self.conn.close()
            logger.info("Index closed successfully")
        except Exception as e:
            logger.error(f"Error closing index: {str(e)}")


class IndexManager:
    """
    High-level manager for document indexing operations
    Provides convenient methods for common indexing tasks
    """
    
    def __init__(self, index: DocumentIndex):
        """Initialize with a document index"""
        self.index = index
    
    def index_directory(self, directory_path: str, file_extensions: Set[str] = None) -> Dict[str, bool]:
        """
        Index all documents in a directory
        
        Args:
            directory_path: Path to directory containing documents
            file_extensions: Set of file extensions to process (e.g., {'.txt', '.pdf'})
            
        Returns:
            Dictionary of {file_path: success_status}
        """
        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"Directory not found: {directory_path}")
            return {}
        
        # Default file extensions
        if file_extensions is None:
            file_extensions = SUPPORTED_EXTENSIONS
        
        results = {}
        
        # Find all files with supported extensions
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                try:
                    # Generate document ID from file path
                    doc_id = self._generate_doc_id(file_path)
                    
                    # Read file content (simplified - in real implementation would use DocumentProcessor)
                    content = self._read_file_content(file_path)
                    
                    if content:
                        success = self.index.add_document(doc_id, str(file_path), content)
                        results[str(file_path)] = success
                    else:
                        results[str(file_path)] = False
                        logger.warning(f"No content extracted from {file_path}")
                        
                except Exception as e:
                    logger.error(f"Error indexing {file_path}: {str(e)}")
                    results[str(file_path)] = False
        
        # Save index after batch operation
        self.index.save_index()
        
        success_count = sum(results.values())
        logger.info(f"Indexed {success_count}/{len(results)} files from {directory_path}")
        
        return results
    
    def _generate_doc_id(self, file_path: Path) -> str:
        """Generate a unique document ID from file path"""
        # Use file name without extension + hash of full path for uniqueness
        name_part = file_path.stem
        path_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
        return f"{name_part}_{path_hash}"
    
    def _read_file_content(self, file_path: Path) -> str:
        """Read file content - simplified implementation"""
        try:
            # For now, only handle text files
            if file_path.suffix.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            else:
                logger.warning(f"Unsupported file type: {file_path.suffix}")
                return ""
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
            return ""
    
    def reindex_document(self, document_id: str, new_content: str = None, 
                        file_path: str = None) -> bool:
        """
        Reindex an existing document with updated content
        
        Args:
            document_id: ID of document to reindex
            new_content: New content (if not provided, will re-read from file)
            file_path: File path (if content needs to be re-read)
            
        Returns:
            Success status
        """
        try:
            # Get existing metadata
            metadata = self.index.get_document_metadata(document_id)
            if not metadata:
                logger.error(f"Document {document_id} not found")
                return False
            
            # Remove existing document
            self.index.remove_document(document_id)
            
            # Get content
            if new_content is None:
                if file_path is None:
                    file_path = metadata.file_path
                
                new_content = self._read_file_content(Path(file_path))
            
            if not new_content:
                logger.error(f"No content available for reindexing {document_id}")
                return False
            
            # Add updated document
            return self.index.add_document(document_id, file_path or metadata.file_path, new_content)
            
        except Exception as e:
            logger.error(f"Error reindexing document {document_id}: {str(e)}")
            return False
    
    def find_duplicates(self, similarity_threshold: float = 0.9) -> List[List[str]]:
        """
        Find duplicate or near-duplicate documents
        
        Args:
            similarity_threshold: Minimum similarity to consider documents as duplicates
            
        Returns:
            List of document ID groups that are similar
        """
        try:
            duplicates = []
            doc_ids = list(self.index.metadata.keys())
            
            # Build similarity matrix if not available
            if self.index.document_vectors is None:
                self.index.build_vector_index()
            
            # Calculate pairwise similarities
            similarity_matrix = cosine_similarity(self.index.document_vectors)
            
            processed = set()
            
            for i, doc_id_1 in enumerate(doc_ids):
                if doc_id_1 in processed:
                    continue
                
                similar_group = [doc_id_1]
                
                for j, doc_id_2 in enumerate(doc_ids):
                    if i != j and doc_id_2 not in processed:
                        if similarity_matrix[i, j] >= similarity_threshold:
                            similar_group.append(doc_id_2)
                            processed.add(doc_id_2)
                
                if len(similar_group) > 1:
                    duplicates.append(similar_group)
                    processed.add(doc_id_1)
            
            logger.info(f"Found {len(duplicates)} duplicate groups")
            return duplicates
            
        except Exception as e:
            logger.error(f"Error finding duplicates: {str(e)}")
            return []
    
    def cleanup_orphaned_documents(self) -> int:
        """
        Remove documents from index where source files no longer exist
        
        Returns:
            Number of orphaned documents removed
        """
        removed_count = 0
        
        for doc_id, metadata in list(self.index.metadata.items()):
            file_path = Path(metadata.file_path)
            
            if not file_path.exists():
                logger.info(f"Removing orphaned document: {doc_id} (file not found: {file_path})")
                if self.index.remove_document(doc_id):
                    removed_count += 1
        
        if removed_count > 0:
            self.index.save_index()
        
        logger.info(f"Cleaned up {removed_count} orphaned documents")
        return removed_count
    
    def export_index_report(self, output_path: str) -> bool:
        """
        Export comprehensive index report
        
        Args:
            output_path: Path for output report file
            
        Returns:
            Success status
        """
        try:
            stats = self.index.get_index_statistics()
            
            report = {
                'generated_at': datetime.now().isoformat(),
                'index_statistics': stats,
                'documents': []
            }
            
            # Add document details
            for doc_id, metadata in self.index.metadata.items():
                doc_info = metadata.to_dict()
                
                # Add search/indexing specific info
                index_entry = self.index.index.get(doc_id)
                if index_entry:
                    doc_info['unique_terms'] = len(index_entry.terms)
                    doc_info['total_term_frequency'] = sum(index_entry.term_frequencies.values())
                
                report['documents'].append(doc_info)
            
            # Write report
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Index report exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting index report: {str(e)}")
            return False


# Utility functions
def create_document_index(index_dir: str = None, **config) -> DocumentIndex:
    """
    Factory function to create a document index
    
    Args:
        index_dir: Directory for index files
        **config: Configuration parameters
        
    Returns:
        Configured DocumentIndex instance
    """
    return DocumentIndex(index_dir=index_dir, config=config)

def batch_index_documents(index: DocumentIndex, documents: Dict[str, Tuple[str, str]]) -> int:
    """
    Batch index multiple documents
    
    Args:
        index: DocumentIndex instance
        documents: Dict of {doc_id: (file_path, content)}
        
    Returns:
        Number of successfully indexed documents
    """
    success_count = 0
    
    for doc_id, (file_path, content) in documents.items():
        if index.add_document(doc_id, file_path, content):
            success_count += 1
    
    # Save index after batch operation
    index.save_index()
    
    logger.info(f"Batch indexed {success_count}/{len(documents)} documents")
    return success_count

def merge_indices(primary_index: DocumentIndex, secondary_index: DocumentIndex) -> DocumentIndex:
    """
    Merge two document indices
    
    Args:
        primary_index: Primary index (documents will be preserved in case of conflicts)
        secondary_index: Secondary index to merge into primary
        
    Returns:
        Merged index (modifies primary_index in place)
    """
    conflicts = []
    merged_count = 0
    
    for doc_id, metadata in secondary_index.metadata.items():
        if doc_id not in primary_index.metadata:
            # Get index entry
            index_entry = secondary_index.index.get(doc_id)
            if index_entry:
                # Add to primary index
                primary_index.metadata[doc_id] = metadata
                primary_index.index[doc_id] = index_entry
                
                # Update inverted index
                for term in index_entry.terms:
                    primary_index.inverted_index[term].add(doc_id)
                
                merged_count += 1
        else:
            conflicts.append(doc_id)
    
    # Update statistics
    primary_index.stats['total_documents'] = len(primary_index.metadata)
    primary_index.stats['total_terms'] = len(primary_index.inverted_index)
    primary_index.stats['last_updated'] = datetime.now()
    
    logger.info(f"Merged {merged_count} documents, {len(conflicts)} conflicts skipped")
    
    return primary_index

def create_index_from_directory(directory_path: str, index_dir: str = None) -> DocumentIndex:
    """
    Create a new index from all documents in a directory
    
    Args:
        directory_path: Path to directory containing documents
        index_dir: Directory to store index files
        
    Returns:
        New DocumentIndex with documents from directory
    """
    # Create new index
    index = create_document_index(index_dir)
    
    # Create manager and index directory
    manager = IndexManager(index)
    results = manager.index_directory(directory_path)
    
    success_count = sum(results.values())
    logger.info(f"Created index with {success_count} documents from {directory_path}")
    
    return index
