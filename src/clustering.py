"""
Document clustering module for grouping similar documents
Supports threshold-based clustering and incremental updates
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, Any
import logging
from collections import defaultdict
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import DBSCAN
import pickle
from pathlib import Path

from config.settings import *

logger = logging.getLogger(__name__)

class DocumentCluster:
    """
    Represents a cluster of similar documents
    """
    
    def __init__(self, cluster_id: str, documents: List[str] = None):
        """
        Initialize document cluster
        
        Args:
            cluster_id: Unique identifier for cluster
            documents: List of document IDs in cluster
        """
        self.cluster_id = cluster_id
        self.documents = documents or []
        self.centroid_doc = None
        self.created_at = None
        self.updated_at = None
        self.similarity_scores = {}  # {doc_id: similarity_to_centroid}
        self.metadata = {}
    
    def add_document(self, doc_id: str, similarity_score: float = None):
        """Add document to cluster"""
        if doc_id not in self.documents:
            self.documents.append(doc_id)
            if similarity_score is not None:
                self.similarity_scores[doc_id] = similarity_score
    
    def remove_document(self, doc_id: str):
        """Remove document from cluster"""
        if doc_id in self.documents:
            self.documents.remove(doc_id)
            self.similarity_scores.pop(doc_id, None)
    
    def get_size(self) -> int:
        """Get number of documents in cluster"""
        return len(self.documents)
    
    def get_average_similarity(self) -> float:
        """Get average similarity score in cluster"""
        if not self.similarity_scores:
            return 0.0
        return sum(self.similarity_scores.values()) / len(self.similarity_scores)
    
    def to_dict(self) -> Dict:
        """Convert cluster to dictionary representation"""
        return {
            'cluster_id': self.cluster_id,
            'documents': self.documents,
            'centroid_doc': self.centroid_doc,
            'size': self.get_size(),
            'average_similarity': self.get_average_similarity(),
            'similarity_scores': self.similarity_scores,
            'metadata': self.metadata
        }

class DocumentClustering:
    """
    Main clustering engine for grouping similar documents
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize document clustering engine
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.clustering_config = CLUSTERING
        
        # Clustering state
        self.clusters = {}  # {cluster_id: DocumentCluster}
        self.document_to_cluster = {}  # {doc_id: cluster_id}
        self.similarity_matrix = None
        self.document_ids = []
        
        # Graph for connected components
        self.similarity_graph = nx.Graph()
        
        # Statistics
        self.cluster_stats = {
            'total_clusters': 0,
            'total_documents': 0,
            'clustered_documents': 0,
            'singleton_clusters': 0,
            'largest_cluster_size': 0,
            'average_cluster_size': 0.0
        }
    
    def threshold_clustering(self, similarity_matrix: np.ndarray, 
                           document_ids: List[str],
                           threshold: float = 0.7) -> Dict[str, DocumentCluster]:
        """
        Perform threshold-based clustering
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            document_ids: List of document IDs
            threshold: Similarity threshold for clustering
            
        Returns:
            Dictionary of clusters
        """
        self.similarity_matrix = similarity_matrix
        self.document_ids = document_ids
        
        # Clear existing clusters
        self.clusters = {}
        self.document_to_cluster = {}
        self.similarity_graph.clear()
        
        # Build similarity graph
        n_docs = len(document_ids)
        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                if similarity_matrix[i, j] >= threshold:
                    self.similarity_graph.add_edge(
                        document_ids[i], 
                        document_ids[j],
                        weight=similarity_matrix[i, j]
                    )
        
        # Find connected components (clusters)
        connected_components = list(nx.connected_components(self.similarity_graph))
        
        # Create clusters from connected components
        for cluster_idx, component in enumerate(connected_components):
            cluster_id = f"cluster_{cluster_idx}"
            
            # Only create cluster if it has minimum size
            if len(component) >= self.clustering_config["min_cluster_size"]:
                cluster = DocumentCluster(cluster_id, list(component))
                
                # Calculate similarity scores for each document in cluster
                for doc_id in component:
                    doc_idx = document_ids.index(doc_id)
                    
                    # Calculate average similarity to other documents in cluster
                    similarities = []
                    for other_doc_id in component:
                        if other_doc_id != doc_id:
                            other_idx = document_ids.index(other_doc_id)
                            similarities.append(similarity_matrix[doc_idx, other_idx])
                    
                    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
                    cluster.similarity_scores[doc_id] = avg_similarity
                
                # Set centroid as document with highest average similarity
                if cluster.similarity_scores:
                    cluster.centroid_doc = max(
                        cluster.similarity_scores,
                        key=cluster.similarity_scores.get
                    )
                
                self.clusters[cluster_id] = cluster
                
                # Update document to cluster mapping
                for doc_id in component:
                    self.document_to_cluster[doc_id] = cluster_id
        
        # Update statistics
        self._update_cluster_stats()
        
        logger.info(f"Created {len(self.clusters)} clusters from {n_docs} documents")
        return self.clusters
    
    def hierarchical_clustering(self, similarity_matrix: np.ndarray,
                              document_ids: List[str],
                              max_clusters: int = None) -> Dict[str, DocumentCluster]:
        """
        Perform hierarchical clustering
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            document_ids: List of document IDs
            max_clusters: Maximum number of clusters
            
        Returns:
            Dictionary of clusters
        """
        # Convert similarity to distance matrix
        distance_matrix = 1 - similarity_matrix
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(distance_matrix, method='average')
        
        # Determine number of clusters
        if max_clusters is None:
            max_clusters = min(len(document_ids) // 2, 10)
        
        # Get cluster labels
        cluster_labels = fcluster(linkage_matrix, max_clusters, criterion='maxclust')
        
        # Create clusters
        clusters = {}
        for cluster_id in range(1, max_clusters + 1):
            cluster_docs = [
                document_ids[i] for i, label in enumerate(cluster_labels)
                if label == cluster_id
            ]
            
            if len(cluster_docs) >= self.clustering_config["min_cluster_size"]:
                cluster = DocumentCluster(f"hier_cluster_{cluster_id}", cluster_docs)
                clusters[cluster.cluster_id] = cluster
        
        self.clusters = clusters
        self._update_cluster_stats()
        
        return clusters
    
    def dbscan_clustering(self, similarity_matrix: np.ndarray,
                         document_ids: List[str],
                         eps: float = 0.3,
                         min_samples: int = 2) -> Dict[str, DocumentCluster]:
        """
        Perform DBSCAN clustering
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            document_ids: List of document IDs
            eps: Maximum distance between samples
            min_samples: Minimum samples in neighborhood
            
        Returns:
            Dictionary of clusters
        """
        # Convert similarity to distance matrix
        distance_matrix = 1 - similarity_matrix
        
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        cluster_labels = dbscan.fit_predict(distance_matrix)
        
        # Create clusters
        clusters = {}
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Noise points
                continue
            
            cluster_docs = [
                document_ids[i] for i, label in enumerate(cluster_labels)
                if label == cluster_id
            ]
            
            if len(cluster_docs) >= self.clustering_config["min_cluster_size"]:
                cluster = DocumentCluster(f"dbscan_cluster_{cluster_id}", cluster_docs)
                clusters[cluster.cluster_id] = cluster
        
        self.clusters = clusters
        self._update_cluster_stats()
        
        return clusters
    
    def incremental_clustering(self, new_doc_id: str, 
                             similarities: Dict[str, float],
                             threshold: float = 0.7) -> Optional[str]:
        """
        Add new document to existing clusters incrementally
        
        Args:
            new_doc_id: ID of new document
            similarities: Similarities to existing documents
            threshold: Similarity threshold
            
        Returns:
            Cluster ID if document was added to cluster, None otherwise
        """
        best_cluster = None
        best_similarity = 0.0
        
        # Find best matching cluster
        for cluster_id, cluster in self.clusters.items():
            cluster_similarities = []
            
            for doc_id in cluster.documents:
                if doc_id in similarities:
                    cluster_similarities.append(similarities[doc_id])
            
            if cluster_similarities:
                avg_similarity = sum(cluster_similarities) / len(cluster_similarities)
                
                if avg_similarity >= threshold and avg_similarity > best_similarity:
                    best_similarity = avg_similarity
                    best_cluster = cluster_id
        
        # Add to best cluster or create new one
        if best_cluster:
            self.clusters[best_cluster].add_document(new_doc_id, best_similarity)
            self.document_to_cluster[new_doc_id] = best_cluster
            logger.info(f"Added {new_doc_id} to {best_cluster}")
            return best_cluster
        else:
            # Check if we should create a new cluster with similar documents
            similar_docs = [
                doc_id for doc_id, sim in similarities.items()
                if sim >= threshold
            ]
            
            if len(similar_docs) >= self.clustering_config["min_cluster_size"] - 1:
                # Create new cluster
                new_cluster_id = f"cluster_{len(self.clusters)}"
                new_cluster = DocumentCluster(new_cluster_id, [new_doc_id] + similar_docs)
                
                self.clusters[new_cluster_id] = new_cluster
                self.document_to_cluster[new_doc_id] = new_cluster_id
                
                # Update mappings for similar documents
                for doc_id in similar_docs:
                    if doc_id in self.document_to_cluster:
                        old_cluster_id = self.document_to_cluster[doc_id]
                        self.clusters[old_cluster_id].remove_document(doc_id)
                    
                    self.document_to_cluster[doc_id] = new_cluster_id
                
                logger.info(f"Created new cluster {new_cluster_id} with {len(similar_docs) + 1} documents")
                return new_cluster_id
        
        return None
    
    def merge_clusters(self, cluster_id1: str, cluster_id2: str) -> str:
        """
        Merge two clusters
        
        Args:
            cluster_id1: First cluster ID
            cluster_id2: Second cluster ID
            
        Returns:
            ID of merged cluster
        """
        if cluster_id1 not in self.clusters or cluster_id2 not in self.clusters:
            raise ValueError("One or both clusters not found")
        
        cluster1 = self.clusters[cluster_id1]
        cluster2 = self.clusters[cluster_id2]
        
        # Merge documents
        merged_docs = cluster1.documents + cluster2.documents
        merged_cluster = DocumentCluster(cluster_id1, merged_docs)
        
        # Merge similarity scores
        merged_cluster.similarity_scores.update(cluster1.similarity_scores)
        merged_cluster.similarity_scores.update(cluster2.similarity_scores)
        
        # Update mappings
        for doc_id in cluster2.documents:
            self.document_to_cluster[doc_id] = cluster_id1
        
        # Remove old cluster
        del self.clusters[cluster_id2]
        self.clusters[cluster_id1] = merged_cluster
        
        logger.info(f"Merged {cluster_id2} into {cluster_id1}")
        return cluster_id1
    
    def split_cluster(self, cluster_id: str, threshold: float = 0.5) -> List[str]:
        """
        Split a cluster based on internal similarities
        
        Args:
            cluster_id: Cluster ID to split
            threshold: Similarity threshold for splitting
            
        Returns:
            List of new cluster IDs
        """
        if cluster_id not in self.clusters:
            raise ValueError(f"Cluster {cluster_id} not found")
        
        cluster = self.clusters[cluster_id]
        
        if len(cluster.documents) <= self.clustering_config["min_cluster_size"]:
            logger.warning(f"Cluster {cluster_id} too small to split")
            return [cluster_id]
        
        # Get document indices
        doc_indices = [self.document_ids.index(doc_id) for doc_id in cluster.documents]
        
        # Extract sub-similarity matrix
        sub_similarity_matrix = self.similarity_matrix[np.ix_(doc_indices, doc_indices)]
        
        # Re-cluster using threshold
        temp_clustering = DocumentClustering()
        sub_clusters = temp_clustering.threshold_clustering(
            sub_similarity_matrix, cluster.documents, threshold
        )
        
        # Create new clusters
        new_cluster_ids = []
        for i, (_, sub_cluster) in enumerate(sub_clusters.items()):
            new_cluster_id = f"{cluster_id}_split_{i}"
            new_cluster = DocumentCluster(new_cluster_id, sub_cluster.documents)
            new_cluster.similarity_scores = sub_cluster.similarity_scores
            
            self.clusters[new_cluster_id] = new_cluster
            new_cluster_ids.append(new_cluster_id)
            
            # Update mappings
            for doc_id in sub_cluster.documents:
                self.document_to_cluster[doc_id] = new_cluster_id
        
        # Remove original cluster
        del self.clusters[cluster_id]
        
        logger.info(f"Split {cluster_id} into {len(new_cluster_ids)} clusters")
        return new_cluster_ids
    
    def get_cluster_for_document(self, doc_id: str) -> Optional[str]:
        """Get cluster ID for a document"""
        return self.document_to_cluster.get(doc_id)
    
    def get_documents_in_cluster(self, cluster_id: str) -> List[str]:
        """Get all documents in a cluster"""
        if cluster_id in self.clusters:
            return self.clusters[cluster_id].documents
        return []
    
    def get_cluster_summary(self, cluster_id: str) -> Dict:
        """Get summary information for a cluster"""
        if cluster_id not in self.clusters:
            return {}
        
        cluster = self.clusters[cluster_id]
        return cluster.to_dict()
    
    def _update_cluster_stats(self):
        """Update clustering statistics"""
        if not self.clusters:
            return
        
        cluster_sizes = [cluster.get_size() for cluster in self.clusters.values()]
        
        self.cluster_stats = {
            'total_clusters': len(self.clusters),
            'total_documents': len(self.document_ids),
            'clustered_documents': sum(cluster_sizes),
            'singleton_clusters': sum(1 for size in cluster_sizes if size == 1),
            'largest_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'average_cluster_size': sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0
        }
    
    def get_clustering_stats(self) -> Dict:
        """Get clustering statistics"""
        return self.cluster_stats
    
    def save_clusters(self, filepath: str):
        """Save clusters to file"""
        cluster_data = {
            'clusters': {cid: cluster.to_dict() for cid, cluster in self.clusters.items()},
            'document_to_cluster': self.document_to_cluster,
            'cluster_stats': self.cluster_stats,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(cluster_data, f)
        
        logger.info(f"Clusters saved to {filepath}")
    
    def load_clusters(self, filepath: str):
        """Load clusters from file"""
        with open(filepath, 'rb') as f:
            cluster_data = pickle.load(f)
        
        # Reconstruct clusters
        self.clusters = {}
        for cid, cluster_dict in cluster_data['clusters'].items():
            cluster = DocumentCluster(cid, cluster_dict['documents'])
            cluster.centroid_doc = cluster_dict.get('centroid_doc')
            cluster.similarity_scores = cluster_dict.get('similarity_scores', {})
            cluster.metadata = cluster_dict.get('metadata', {})
            self.clusters[cid] = cluster
        
        self.document_to_cluster = cluster_data['document_to_cluster']
        self.cluster_stats = cluster_data['cluster_stats']
        self.config = cluster_data.get('config', {})
        
        logger.info(f"Clusters loaded from {filepath}")
    
    def export_clusters_to_csv(self, filepath: str):
        """Export clusters to CSV format"""
        data = []
        
        for cluster_id, cluster in self.clusters.items():
            for doc_id in cluster.documents:
                data.append({
                    'cluster_id': cluster_id,
                    'document_id': doc_id,
                    'cluster_size': cluster.get_size(),
                    'similarity_score': cluster.similarity_scores.get(doc_id, 0.0),
                    'is_centroid': doc_id == cluster.centroid_doc
                })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        logger.info(f"Clusters exported to {filepath}")
    
    def get_cluster_network(self) -> nx.Graph:
        """Get network representation of clusters"""
        return self.similarity_graph.copy()