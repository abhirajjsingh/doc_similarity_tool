"""
Alert system for new document similarity detection
Provides real-time similarity checking and notifications
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np

from similarity_engine import SimilarityEngine
from clustering import DocumentClustering
from document_processor import DocumentProcessor
from config.settings import *

logger = logging.getLogger(__name__)

@dataclass
class SimilarityAlert:
    """
    Represents a similarity alert for a new document
    """
    document_id: str
    similar_documents: List[Tuple[str, float]]
    cluster_id: Optional[str] = None
    alert_type: str
    timestamp: datetime = None
    severity: str = "medium"  # low, medium, high, critical
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize fields after object creation"""
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class AlertConfig:
    """Configuration for alert system"""
    similarity_threshold: float = 0.7
    max_alerts_per_hour: int = 100
    enable_clustering: bool = True
    enable_notifications: bool = True
    alert_types: List[str] = None
    output_file: Optional[str] = None
    
    def __post_init__(self):
        if self.alert_types is None:
            self.alert_types = ["high_similarity", "cluster_addition", "duplicate_detection"]

class AlertSystem:
    """
    Real-time alert system for document similarity detection
    Monitors new documents and generates alerts for similar content
    """
    
    def __init__(self, similarity_engine: SimilarityEngine,
                 clustering_engine: DocumentClustering = None,
                 document_processor: DocumentProcessor = None,
                 config: AlertConfig = None):
        """
        Initialize alert system
        
        Args:
            similarity_engine: Engine for computing similarities
            clustering_engine: Optional clustering engine
            document_processor: Optional document processor
            config: Alert configuration
        """
        self.similarity_engine = similarity_engine
        self.clustering_engine = clustering_engine
        self.document_processor = document_processor
        self.config = config or AlertConfig()
        
        # Alert storage and tracking
        self.alerts: List[SimilarityAlert] = []
        self.alert_history: Dict[str, List[SimilarityAlert]] = {}
        self.processed_documents: set = set()
        
        # Performance tracking
        self.alert_stats = {
            'total_alerts': 0,
            'alerts_by_type': {},
            'alerts_by_severity': {},
            'processing_times': [],
            'last_alert_time': None
        }
        
        # Rate limiting
        self.alerts_last_hour = []
        
        logger.info(f"Alert system initialized with threshold {self.config.similarity_threshold}")
    
    def process_new_document(self, document_id: str, document_text: str = None,
                           file_path: str = None) -> Optional[SimilarityAlert]:
        """
        Process a new document and generate alerts if similar documents are found
        
        Args:
            document_id: Unique identifier for the document
            document_text: Text content of the document
            file_path: Optional file path if text needs to be extracted
            
        Returns:
            SimilarityAlert if similarities found, None otherwise
        """
        start_time = datetime.now()
        
        try:
            # Extract text if not provided
            if document_text is None and file_path is not None:
                if self.document_processor is None:
                    raise ValueError("Document processor required for file processing")
                document_text = self.document_processor.extract_text(file_path)
            
            if not document_text:
                logger.warning(f"No text content found for document {document_id}")
                return None
            
            # Check if document already processed
            if document_id in self.processed_documents:
                logger.info(f"Document {document_id} already processed")
                return None
            
            # Find similar documents
            similar_docs = self._find_similar_documents(document_id, document_text)
            
            if not similar_docs:
                logger.info(f"No similar documents found for {document_id}")
                self.processed_documents.add(document_id)
                return None
            
            # Determine alert type and severity
            alert_type, severity = self._classify_alert(similar_docs)
            
            # Check for clustering
            cluster_id = None
            if self.clustering_engine and self.config.enable_clustering:
                cluster_id = self._check_clustering(document_id, similar_docs)
            
            # Create alert
            alert = SimilarityAlert(
                document_id=document_id,
                similar_documents=similar_docs,
                cluster_id=cluster_id,
                alert_type=alert_type,
                severity=severity,
                metadata={
                    'document_length': len(document_text),
                    'processing_time': (datetime.now() - start_time).total_seconds(),
                    'similarity_count': len(similar_docs)
                }
            )
            
            # Store alert
            self._store_alert(alert)
            self.processed_documents.add(document_id)
            
            # Update statistics
            self._update_stats(alert, start_time)
            
            logger.info(f"Alert generated for document {document_id}: {alert_type} ({severity})")
            return alert
            
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {str(e)}")
            return None
    
    def _find_similar_documents(self, document_id: str, 
                              document_text: str) -> List[Tuple[str, float]]:
        """Find documents similar to the given document"""
        try:
            # Use LSH if available for efficiency
            if hasattr(self.similarity_engine, 'lsh_index') and self.similarity_engine.lsh_index:
                similar_docs = self.similarity_engine.find_similar_documents_lsh(
                    document_text, top_k=10
                )
            else:
                # Fallback to standard similarity computation
                similar_docs = self._compute_standard_similarity(document_text)
            
            # Filter by threshold
            filtered_docs = [
                (doc_id, score) for doc_id, score in similar_docs
                if score >= self.config.similarity_threshold
            ]
            
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {str(e)}")
            return []
    
    def _compute_standard_similarity(self, document_text: str) -> List[Tuple[str, float]]:
        """Compute similarity using standard methods when LSH is not available"""
        similarities = []
        
        try:
            # If we have a pre-computed similarity matrix, use it
            if hasattr(self.similarity_engine, 'document_texts') and self.similarity_engine.document_texts:
                # Get document vectors/embeddings if available
                if hasattr(self.similarity_engine, 'compute_similarity_score'):
                    for doc_id, existing_text in self.similarity_engine.document_texts.items():
                        if existing_text:  # Skip empty documents
                            similarity = self.similarity_engine.compute_similarity_score(
                                document_text, existing_text
                            )
                            if similarity > 0:
                                similarities.append((doc_id, similarity))
                else:
                    logger.warning("Similarity engine doesn't have compute_similarity_score method")
            else:
                logger.warning("No pre-loaded documents available for comparison")
                
        except Exception as e:
            logger.error(f"Error in standard similarity computation: {str(e)}")
        
        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    
    def _classify_alert(self, similar_docs: List[Tuple[str, float]]) -> Tuple[str, str]:
        """Classify alert type and severity based on similarity scores"""
        max_similarity = max(score for _, score in similar_docs) if similar_docs else 0
        similarity_count = len(similar_docs)
        
        # Determine alert type
        if max_similarity >= 0.95:
            alert_type = "duplicate_detection"
        elif max_similarity >= 0.8:
            alert_type = "high_similarity"
        else:
            alert_type = "moderate_similarity"
        
        # Determine severity
        if max_similarity >= 0.95 or similarity_count >= 10:
            severity = "critical"
        elif max_similarity >= 0.85 or similarity_count >= 5:
            severity = "high"
        elif max_similarity >= 0.75 or similarity_count >= 3:
            severity = "medium"
        else:
            severity = "low"
        
        return alert_type, severity
    
    def _check_clustering(self, document_id: str, 
                         similar_docs: List[Tuple[str, float]]) -> Optional[str]:
        """Check if document should be added to an existing cluster"""
        try:
            # Find clusters for similar documents
            clusters = set()
            for doc_id, _ in similar_docs:
                cluster = self.clustering_engine.get_cluster_for_document(doc_id)
                if cluster:
                    clusters.add(cluster)
            
            # If similar documents belong to a single cluster, add to that cluster
            if len(clusters) == 1:
                cluster_id = list(clusters)[0]
                # Add document to cluster (this would need to be implemented)
                logger.info(f"Adding document {document_id} to cluster {cluster_id}")
                return cluster_id
            elif len(clusters) > 1:
                # Multiple clusters - might need merging logic
                logger.info(f"Document {document_id} similar to multiple clusters: {clusters}")
                return None
            else:
                # No existing clusters - might create new one
                return None
                
        except Exception as e:
            logger.error(f"Error checking clustering: {str(e)}")
            return None
    
    def _store_alert(self, alert: SimilarityAlert):
        """Store alert in memory and optionally to file"""
        self.alerts.append(alert)
        
        # Store in history by document
        if alert.document_id not in self.alert_history:
            self.alert_history[alert.document_id] = []
        self.alert_history[alert.document_id].append(alert)
        
        # Rate limiting
        self.alerts_last_hour.append(alert.timestamp)
        # Remove alerts older than 1 hour
        cutoff_time = datetime.now() - datetime.timedelta(hours=1)
        self.alerts_last_hour = [
            ts for ts in self.alerts_last_hour if ts > cutoff_time
        ]
        
        # Save to file if configured
        if self.config.output_file:
            self._save_alert_to_file(alert)
    
    def _save_alert_to_file(self, alert: SimilarityAlert):
        """Save alert to JSON file"""
        try:
            alert_data = {
                'document_id': alert.document_id,
                'similar_documents': alert.similar_documents,
                'cluster_id': alert.cluster_id,
                'alert_type': alert.alert_type,
                'timestamp': alert.timestamp.isoformat(),
                'severity': alert.severity,
                'metadata': alert.metadata
            }
            
            output_path = Path(self.config.output_file)
            
            # Append to existing file or create new
            alerts_data = []
            if output_path.exists():
                with open(output_path, 'r') as f:
                    alerts_data = json.load(f)
            
            alerts_data.append(alert_data)
            
            with open(output_path, 'w') as f:
                json.dump(alerts_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving alert to file: {str(e)}")
    
    def _update_stats(self, alert: SimilarityAlert, start_time: datetime):
        """Update performance and usage statistics"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        self.alert_stats['total_alerts'] += 1
        self.alert_stats['last_alert_time'] = alert.timestamp
        self.alert_stats['processing_times'].append(processing_time)
        
        # Update type counts
        if alert.alert_type not in self.alert_stats['alerts_by_type']:
            self.alert_stats['alerts_by_type'][alert.alert_type] = 0
        self.alert_stats['alerts_by_type'][alert.alert_type] += 1
        
        # Update severity counts
        if alert.severity not in self.alert_stats['alerts_by_severity']:
            self.alert_stats['alerts_by_severity'][alert.severity] = 0
        self.alert_stats['alerts_by_severity'][alert.severity] += 1
        
        # Keep only last 1000 processing times to prevent memory issues
        if len(self.alert_stats['processing_times']) > 1000:
            self.alert_stats['processing_times'] = self.alert_stats['processing_times'][-1000:]
    
    def get_alerts_for_document(self, document_id: str) -> List[SimilarityAlert]:
        """Get all alerts for a specific document"""
        return self.alert_history.get(document_id, [])
    
    def get_recent_alerts(self, hours: int = 24) -> List[SimilarityAlert]:
        """Get alerts from the last N hours"""
        cutoff_time = datetime.now() - datetime.timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp > cutoff_time]
    
    def get_alerts_by_severity(self, severity: str) -> List[SimilarityAlert]:
        """Get alerts filtered by severity level"""
        return [alert for alert in self.alerts if alert.severity == severity]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive alert system statistics"""
        stats = self.alert_stats.copy()
        
        # Add derived statistics
        if self.alert_stats['processing_times']:
            stats['avg_processing_time'] = np.mean(self.alert_stats['processing_times'])
            stats['max_processing_time'] = max(self.alert_stats['processing_times'])
            stats['min_processing_time'] = min(self.alert_stats['processing_times'])
        
        stats['alerts_last_hour'] = len(self.alerts_last_hour)
        stats['rate_limit_reached'] = len(self.alerts_last_hour) >= self.config.max_alerts_per_hour
        stats['total_documents_processed'] = len(self.processed_documents)
        
        return stats
    
    def is_rate_limited(self) -> bool:
        """Check if rate limit has been reached"""
        return len(self.alerts_last_hour) >= self.config.max_alerts_per_hour
    
    def clear_alerts(self):
        """Clear all stored alerts (use with caution)"""
        self.alerts.clear()
        self.alert_history.clear()
        self.alert_stats = {
            'total_alerts': 0,
            'alerts_by_type': {},
            'alerts_by_severity': {},
            'processing_times': [],
            'last_alert_time': None
        }
        logger.info("All alerts cleared")
    
    def export_alerts(self, file_path: str, format: str = "json") -> bool:
        """
        Export all alerts to file
        
        Args:
            file_path: Output file path
            format: Export format ("json" or "csv")
            
        Returns:
            Success status
        """
        try:
            if format.lower() == "json":
                alerts_data = []
                for alert in self.alerts:
                    alert_data = {
                        'document_id': alert.document_id,
                        'similar_documents': alert.similar_documents,
                        'cluster_id': alert.cluster_id,
                        'alert_type': alert.alert_type,
                        'timestamp': alert.timestamp.isoformat(),
                        'severity': alert.severity,
                        'metadata': alert.metadata
                    }
                    alerts_data.append(alert_data)
                
                with open(file_path, 'w') as f:
                    json.dump(alerts_data, f, indent=2)
                    
            elif format.lower() == "csv":
                import csv
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['document_id', 'alert_type', 'severity', 'timestamp', 
                                   'similar_count', 'cluster_id', 'max_similarity'])
                    
                    for alert in self.alerts:
                        max_sim = max(score for _, score in alert.similar_documents) if alert.similar_documents else 0
                        writer.writerow([
                            alert.document_id,
                            alert.alert_type,
                            alert.severity,
                            alert.timestamp.isoformat(),
                            len(alert.similar_documents),
                            alert.cluster_id or '',
                            f"{max_sim:.3f}"
                        ])
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Alerts exported to {file_path} in {format} format")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting alerts: {str(e)}")
            return False
    
    def batch_process_files(self, file_paths: List[str]) -> List[SimilarityAlert]:
        """
        Process multiple files in batch and return all generated alerts
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of generated alerts
        """
        alerts = []
        
        for file_path in file_paths:
            try:
                # Generate document ID from file path
                doc_id = Path(file_path).stem
                
                # Process document
                alert = self.process_new_document(
                    document_id=doc_id,
                    file_path=file_path
                )
                
                if alert:
                    alerts.append(alert)
                    
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                continue
        
        logger.info(f"Batch processing complete: {len(alerts)} alerts generated from {len(file_paths)} files")
        return alerts
    
    def set_alert_thresholds(self, similarity_threshold: float = None,
                           severity_thresholds: Dict[str, float] = None):
        """
        Update alert thresholds dynamically
        
        Args:
            similarity_threshold: New similarity threshold for alerts
            severity_thresholds: New thresholds for severity classification
        """
        if similarity_threshold is not None:
            self.config.similarity_threshold = similarity_threshold
            logger.info(f"Updated similarity threshold to {similarity_threshold}")
        
        if severity_thresholds is not None:
            # Store custom severity thresholds
            self.severity_thresholds = severity_thresholds
            logger.info(f"Updated severity thresholds: {severity_thresholds}")
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of all alerts
        
        Returns:
            Dictionary containing alert summary statistics
        """
        if not self.alerts:
            return {
                'total_alerts': 0,
                'summary': 'No alerts generated yet'
            }
        
        # Calculate summary statistics
        severity_counts = {}
        type_counts = {}
        recent_alerts = 0
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        for alert in self.alerts:
            # Count by severity
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
            
            # Count by type
            type_counts[alert.alert_type] = type_counts.get(alert.alert_type, 0) + 1
            
            # Count recent alerts
            if alert.timestamp > cutoff_time:
                recent_alerts += 1
        
        # Find most similar documents
        max_similarity = 0
        most_similar_pair = None
        
        for alert in self.alerts:
            if alert.similar_documents:
                alert_max = max(score for _, score in alert.similar_documents)
                if alert_max > max_similarity:
                    max_similarity = alert_max
                    most_similar_pair = (alert.document_id, alert.similar_documents[0][0])
        
        return {
            'total_alerts': len(self.alerts),
            'alerts_last_24h': recent_alerts,
            'severity_distribution': severity_counts,
            'type_distribution': type_counts,
            'highest_similarity': max_similarity,
            'most_similar_documents': most_similar_pair,
            'total_documents_processed': len(self.processed_documents),
            'rate_limit_status': 'active' if self.is_rate_limited() else 'normal'
        }
    
    def remove_document_alerts(self, document_id: str) -> int:
        """
        Remove all alerts for a specific document
        
        Args:
            document_id: ID of document to remove alerts for
            
        Returns:
            Number of alerts removed
        """
        # Remove from main alerts list
        initial_count = len(self.alerts)
        self.alerts = [alert for alert in self.alerts if alert.document_id != document_id]
        
        # Remove from history
        if document_id in self.alert_history:
            del self.alert_history[document_id]
        
        # Remove from processed documents set
        self.processed_documents.discard(document_id)
        
        removed_count = initial_count - len(self.alerts)
        logger.info(f"Removed {removed_count} alerts for document {document_id}")
        
        return removed_count
    
    def update_document_clustering(self, document_id: str, new_cluster_id: str):
        """
        Update cluster assignment for a document's alerts
        
        Args:
            document_id: ID of the document
            new_cluster_id: New cluster ID to assign
        """
        updated_count = 0
        
        # Update alerts for this document
        for alert in self.alerts:
            if alert.document_id == document_id:
                alert.cluster_id = new_cluster_id
                updated_count += 1
        
        # Update alert history
        if document_id in self.alert_history:
            for alert in self.alert_history[document_id]:
                alert.cluster_id = new_cluster_id
        
        logger.info(f"Updated cluster assignment for {updated_count} alerts of document {document_id}")
    
    def get_cluster_alerts(self, cluster_id: str) -> List[SimilarityAlert]:
        """
        Get all alerts for documents in a specific cluster
        
        Args:
            cluster_id: ID of the cluster
            
        Returns:
            List of alerts for documents in the cluster
        """
        return [alert for alert in self.alerts if alert.cluster_id == cluster_id]


# Utility functions for alert management
def create_alert_system(similarity_engine: SimilarityEngine,
                       clustering_engine: DocumentClustering = None,
                       document_processor: DocumentProcessor = None,
                       **config_kwargs) -> AlertSystem:
    """
    Factory function to create a configured alert system
    
    Args:
        similarity_engine: Pre-configured similarity engine
        clustering_engine: Optional clustering engine
        document_processor: Optional document processor
        **config_kwargs: Configuration parameters for AlertConfig
        
    Returns:
        Configured AlertSystem instance
    """
    config = AlertConfig(**config_kwargs)
    return AlertSystem(
        similarity_engine=similarity_engine,
        clustering_engine=clustering_engine,
        document_processor=document_processor,
        config=config
    )

def batch_process_documents(alert_system: AlertSystem,
                          documents: Dict[str, str]) -> List[SimilarityAlert]:
    """
    Process multiple documents and return all generated alerts
    
    Args:
        alert_system: Configured alert system
        documents: Dictionary of {document_id: document_text}
        
    Returns:
        List of generated alerts
    """
    alerts = []
    
    for doc_id, doc_text in documents.items():
        alert = alert_system.process_new_document(doc_id, doc_text)
        if alert:
            alerts.append(alert)
    
    return alerts