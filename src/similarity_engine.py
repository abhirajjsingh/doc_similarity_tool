"""
Core similarity engine for document comparison
Implements multiple similarity metrics with optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.sparse import csr_matrix
from datasketch import MinHash, MinHashLSH
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging
from pathlib import Path
import pickle
import joblib
from config.settings import *

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

logger = logging.getLogger(__name__)

class SimilarityEngine:
    """
    Core engine for document similarity detection
    Supports multiple similarity metrics and optimization techniques
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the similarity engine with configuration"""
        self.config = config or {}
        self.text_config = TEXT_PROCESSING
        self.lsh_config = LSH_SETTINGS
        self.performance_config = PERFORMANCE
        
        # Initialize components
        self.vectorizer = None
        self.document_vectors = None
        self.document_ids = []
        self.document_texts = {}
        self.lsh_index = None
        self.similarity_matrix = None
        
        # Cache for computed similarities
        self.similarity_cache = {}
        
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for similarity computation
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text or len(text) < self.text_config["min_doc_length"]:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords if configured
        if self.text_config["remove_stopwords"]:
            tokens = word_tokenize(text)
            tokens = [token for token in tokens if token not in self.stop_words]
            text = ' '.join(tokens)
        
        return text
    
    def extract_features(self, documents: List[str], method: str = "tfidf") -> csr_matrix:
        """
        Extract features from documents using specified method
        
        Args:
            documents: List of document texts
            method: Feature extraction method ("tfidf", "count", "binary")
            
        Returns:
            Feature matrix
        """
        if method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=self.text_config["max_features"],
                min_df=self.text_config["min_df"],
                max_df=self.text_config["max_df"],
                ngram_range=self.text_config["ngram_range"],
                stop_words='english' if self.text_config["remove_stopwords"] else None
            )
        elif method == "count":
            self.vectorizer = CountVectorizer(
                max_features=self.text_config["max_features"],
                min_df=self.text_config["min_df"],
                max_df=self.text_config["max_df"],
                ngram_range=self.text_config["ngram_range"],
                stop_words='english' if self.text_config["remove_stopwords"] else None
            )
        else:
            raise ValueError(f"Unsupported feature extraction method: {method}")
        
        # Fit and transform documents
        feature_matrix = self.vectorizer.fit_transform(documents)
        
        return feature_matrix
    
    def compute_similarity_matrix(self, feature_matrix: csr_matrix, 
                                 metric: str = "cosine") -> np.ndarray:
        """
        Compute similarity matrix for all document pairs
        
        Args:
            feature_matrix: Document feature matrix
            metric: Similarity metric ("cosine", "euclidean", "manhattan")
            
        Returns:
            Similarity matrix
        """
        if metric == "cosine":
            similarity_matrix = cosine_similarity(feature_matrix)
        elif metric == "euclidean":
            # Convert distance to similarity (0 = identical, higher = more different)
            distance_matrix = euclidean_distances(feature_matrix)
            # Normalize to [0, 1] similarity scale
            max_distance = np.max(distance_matrix)
            similarity_matrix = 1 - (distance_matrix / max_distance)
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
        
        return similarity_matrix
    
    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Compute Jaccard similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Jaccard similarity score
        """
        tokens1 = set(word_tokenize(text1.lower()))
        tokens2 = set(word_tokenize(text2.lower()))
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union)
    
    def create_minhash(self, text: str, num_perm: int = 128) -> MinHash:
        """
        Create MinHash signature for text
        
        Args:
            text: Text to hash
            num_perm: Number of permutations
            
        Returns:
            MinHash signature
        """
        minhash = MinHash(num_perm=num_perm)
        tokens = word_tokenize(text.lower())
        
        for token in tokens:
            minhash.update(token.encode('utf8'))
        
        return minhash
    
    def build_lsh_index(self, documents: List[str], threshold: float = 0.7):
        """
        Build LSH index for fast approximate similarity search
        
        Args:
            documents: List of document texts
            threshold: LSH threshold for similarity
        """
        self.lsh_index = MinHashLSH(
            threshold=threshold,
            num_perm=self.lsh_config["num_perm"]
        )
        
        # Create MinHash signatures and add to LSH index
        for i, doc in enumerate(documents):
            minhash = self.create_minhash(doc)
            self.lsh_index.insert(f"doc_{i}", minhash)
    
    def find_similar_documents_lsh(self, query_text: str, 
                                  top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find similar documents using LSH index
        
        Args:
            query_text: Query text
            top_k: Number of similar documents to return
            
        Returns:
            List of (document_id, similarity_score) tuples
        """
        if self.lsh_index is None:
            raise ValueError("LSH index not built. Call build_lsh_index first.")
        
        query_minhash = self.create_minhash(query_text)
        similar_docs = self.lsh_index.query(query_minhash)
        
        # Compute exact similarities for LSH candidates
        similarities = []
        for doc_id in similar_docs:
            doc_idx = int(doc_id.split('_')[1])
            if doc_idx < len(self.document_ids):
                doc_text = self.document_texts[self.document_ids[doc_idx]]
                similarity = self.jaccard_similarity(query_text, doc_text)
                similarities.append((doc_id, similarity))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def process_corpus(self, documents: Dict[str, str]) -> Dict[str, Any]:
        """
        Process entire document corpus and build similarity structures
        
        Args:
            documents: Dictionary of {document_id: document_text}
            
        Returns:
            Processing results including similarity matrix and indices
        """
        logger.info(f"Processing corpus with {len(documents)} documents")
        
        # Store document information
        self.document_ids = list(documents.keys())
        self.document_texts = documents
        
        # Preprocess all documents
        preprocessed_docs = [
            self.preprocess_text(text) for text in documents.values()
        ]
        
        # Extract features
        logger.info("Extracting features using TF-IDF")
        self.document_vectors = self.extract_features(
            preprocessed_docs, method="tfidf"
        )
        
        # Compute similarity matrix
        logger.info("Computing similarity matrix")
        self.similarity_matrix = self.compute_similarity_matrix(
            self.document_vectors, metric="cosine"
        )
        
        # Build LSH index for fast similarity search
        logger.info("Building LSH index")
        self.build_lsh_index(preprocessed_docs)
        
        # Find similar pairs above threshold
        similar_pairs = self.find_similar_pairs(
            threshold=SIMILARITY_THRESHOLDS["default"]
        )
        
        results = {
            "num_documents": len(documents),
            "feature_matrix_shape": self.document_vectors.shape,
            "similarity_matrix_shape": self.similarity_matrix.shape,
            "similar_pairs": similar_pairs,
            "num_similar_pairs": len(similar_pairs)
        }
        
        logger.info(f"Found {len(similar_pairs)} similar document pairs")
        return results
    
    def find_similar_pairs(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """
        Find all document pairs above similarity threshold
        
        Args:
            threshold: Similarity threshold
            
        Returns:
            List of (doc1_id, doc2_id, similarity_score) tuples
        """
        if self.similarity_matrix is None:
            raise ValueError("Similarity matrix not computed. Call process_corpus first.")
        
        similar_pairs = []
        n_docs = len(self.document_ids)
        
        # Get upper triangle of similarity matrix (avoid duplicates)
        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                similarity = self.similarity_matrix[i, j]
                if similarity >= threshold:
                    similar_pairs.append((
                        self.document_ids[i],
                        self.document_ids[j],
                        similarity
                    ))
        
        # Sort by similarity score descending
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        return similar_pairs
    
    def find_similar_to_new_document(self, new_doc_text: str, 
                                   threshold: float = 0.7,
                                   top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find documents similar to a new document
        
        Args:
            new_doc_text: Text of new document
            threshold: Similarity threshold
            top_k: Maximum number of similar documents to return
            
        Returns:
            List of (document_id, similarity_score) tuples
        """
        if self.vectorizer is None or self.document_vectors is None:
            raise ValueError("Corpus not processed. Call process_corpus first.")
        
        # Preprocess new document
        preprocessed_text = self.preprocess_text(new_doc_text)
        
        # Transform using existing vectorizer
        new_doc_vector = self.vectorizer.transform([preprocessed_text])
        
        # Compute similarities with all existing documents
        similarities = cosine_similarity(new_doc_vector, self.document_vectors)[0]
        
        # Find documents above threshold
        similar_docs = []
        for i, similarity in enumerate(similarities):
            if similarity >= threshold:
                similar_docs.append((self.document_ids[i], similarity))
        
        # Sort by similarity and return top-k
        similar_docs.sort(key=lambda x: x[1], reverse=True)
        return similar_docs[:top_k]
    
    def save_model(self, filepath: str):
        """Save the similarity engine model"""
        model_data = {
            'vectorizer': self.vectorizer,
            'document_vectors': self.document_vectors,
            'document_ids': self.document_ids,
            'document_texts': self.document_texts,
            'similarity_matrix': self.similarity_matrix,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved similarity engine model"""
        model_data = joblib.load(filepath)
        
        self.vectorizer = model_data['vectorizer']
        self.document_vectors = model_data['document_vectors']
        self.document_ids = model_data['document_ids']
        self.document_texts = model_data['document_texts']
        self.similarity_matrix = model_data['similarity_matrix']
        self.config = model_data['config']
        
        logger.info(f"Model loaded from {filepath}")