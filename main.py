#!/usr/bin/env python3
"""
Document Similarity Detection System - Main Entry Point
========================================================

This is the main application file that demonstrates the complete workflow
of the document similarity detection system including:
- Document processing and text extraction
- Similarity computation and search
- Document clustering
- Real-time alert system
- Document indexing and retrieval

Usage:
    python main.py [command] [options]

Commands:
    demo        - Run interactive demo
    process     - Process documents in a directory
    search      - Search for similar documents
    cluster     - Perform document clustering
    server      - Start web server (future)
    
Examples:
    python main.py demo
    python main.py process --directory data/sample_pdfs
    python main.py search --query "machine learning" --top-k 5
    python main.py cluster --threshold 0.7
"""

import os
import sys
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from document_processor import DocumentProcessor
from similarity_engine import SimilarityEngine
from clustering import DocumentClustering
from indexing import DocumentIndex, IndexManager
from alert_system import AlertSystem, AlertConfig, create_alert_system
from config.settings import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('similarity_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class DocumentSimilarityApp:
    """
    Main application class that orchestrates all components of the 
    document similarity detection system
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the application with all components"""
        self.config = config or {}
        
        # Initialize core components
        logger.info("Initializing Document Similarity Detection System...")
        
        # Document processor for text extraction
        self.processor = DocumentProcessor()
        
        # Similarity engine for computing similarities
        self.similarity_engine = SimilarityEngine()
        
        # Clustering engine for grouping similar documents
        self.clustering_engine = DocumentClustering()
        
        # Document index for fast retrieval
        self.document_index = DocumentIndex()
        self.index_manager = IndexManager(self.document_index)
        
        # Alert system for real-time notifications
        alert_config = AlertConfig(
            similarity_threshold=SIMILARITY_THRESHOLDS["default"],
            enable_clustering=True,
            enable_notifications=True,
            output_file=str(DATA_DIR / "alerts.json")
        )
        
        self.alert_system = create_alert_system(
            similarity_engine=self.similarity_engine,
            clustering_engine=self.clustering_engine,
            document_processor=self.processor,
            **alert_config.__dict__
        )
        
        # Track processed documents
        self.processed_documents = {}
        
        logger.info("System initialization complete!")
    
    def run_demo(self):
        """Run an interactive demonstration of the system"""
        print("\n" + "="*70)
        print("🚀 DOCUMENT SIMILARITY DETECTION SYSTEM DEMO")
        print("="*70)
        
        # Step 1: Process sample documents
        print("\n📁 Step 1: Processing sample documents...")
        self.process_sample_documents()
        
        # Step 2: Build similarity indices
        print("\n🔍 Step 2: Building similarity indices...")
        self.build_indices()
        
        # Step 3: Demonstrate similarity search
        print("\n🎯 Step 3: Demonstrating similarity search...")
        self.demo_similarity_search()
        
        # Step 4: Perform clustering
        print("\n📊 Step 4: Performing document clustering...")
        self.demo_clustering()
        
        # Step 5: Demonstrate alert system
        print("\n🚨 Step 5: Demonstrating alert system...")
        self.demo_alert_system()
        
        # Step 6: Show statistics
        print("\n📈 Step 6: System statistics...")
        self.show_statistics()
        
        print("\n✅ Demo complete! Check the logs and output files for detailed results.")
        print("="*70)
    
    def process_sample_documents(self):
        """Process sample documents in the data directory"""
        sample_dir = SAMPLE_PDFS_DIR
        
        if not sample_dir.exists() or not any(sample_dir.iterdir()):
            print(f"⚠️  No sample documents found in {sample_dir}")
            print("Please add some PDF files to the sample_pdfs directory to run the demo.")
            return
        
        print(f"Processing documents from: {sample_dir}")
        
        # Use index manager to process directory
        results = self.index_manager.index_directory(str(sample_dir))
        
        success_count = sum(results.values())
        total_count = len(results)
        
        print(f"✅ Processed {success_count}/{total_count} documents successfully")
        
        # Store processed documents for later use
        for file_path, success in results.items():
            if success:
                doc_id = Path(file_path).stem
                self.processed_documents[doc_id] = file_path
        
        # Also process any text files in processed_docs
        processed_dir = PROCESSED_DOCS_DIR
        if processed_dir.exists():
            for txt_file in processed_dir.glob("*.txt"):
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    doc_id = txt_file.stem
                    if self.document_index.add_document(doc_id, str(txt_file), content):
                        self.processed_documents[doc_id] = str(txt_file)
                        print(f"✅ Added text document: {doc_id}")
                        
                except Exception as e:
                    print(f"❌ Error processing {txt_file}: {str(e)}")
    
    def build_indices(self):
        """Build similarity indices for fast search"""
        try:
            print("Building TF-IDF vectors...")
            self.document_index.build_vector_index()
            
            print("Building FAISS index (if available)...")
            self.document_index.build_faiss_index()
            
            # Build similarity engine corpus
            if self.processed_documents:
                print("Building similarity engine corpus...")
                documents = {}
                
                for doc_id, file_path in self.processed_documents.items():
                    metadata = self.document_index.get_document_metadata(doc_id)
                    if metadata:
                        # Reconstruct content from index for similarity engine
                        index_entry = self.document_index.index.get(doc_id)
                        if index_entry:
                            # Simple reconstruction from terms (this is approximate)
                            content = " ".join(index_entry.terms)
                            documents[doc_id] = content
                
                if documents:
                    self.similarity_engine.process_corpus(documents)
                    print(f"✅ Built indices for {len(documents)} documents")
                else:
                    print("⚠️  No document content available for similarity engine")
        
        except Exception as e:
            logger.error(f"Error building indices: {str(e)}")
            print(f"❌ Error building indices: {str(e)}")
    
    def demo_similarity_search(self):
        """Demonstrate similarity search capabilities"""
        if not self.processed_documents:
            print("⚠️  No documents available for similarity search")
            return
        
        # Example queries
        test_queries = [
            "machine learning algorithms",
            "data analysis techniques", 
            "artificial intelligence",
            "document processing",
            "research methodology"
        ]
        
        for query in test_queries[:2]:  # Limit to 2 queries for demo
            print(f"\n🔍 Searching for: '{query}'")
            
            try:
                # Try FAISS search first, fallback to regular similarity search
                results = self.document_index.faiss_similarity_search(query, top_k=3)
                
                if not results:
                    results = self.document_index.similarity_search(query, top_k=3)
                
                if results:
                    print("📋 Results:")
                    for doc_id, similarity in results:
                        metadata = self.document_index.get_document_metadata(doc_id)
                        file_name = Path(metadata.file_path).name if metadata else doc_id
                        print(f"  • {file_name}: {similarity:.3f} similarity")
                else:
                    print("  No similar documents found")
                    
            except Exception as e:
                print(f"❌ Search error: {str(e)}")
    
    def demo_clustering(self):
        """Demonstrate document clustering"""
        if len(self.processed_documents) < 2:
            print("⚠️  Need at least 2 documents for clustering")
            return
        
        try:
            # Build similarity matrix for clustering
            if self.document_index.document_vectors is not None:
                print("Computing similarity matrix...")
                
                # Use the document vectors from the index
                from sklearn.metrics.pairwise import cosine_similarity
                similarity_matrix = cosine_similarity(self.document_index.document_vectors)
                
                # Get document IDs
                doc_ids = list(self.processed_documents.keys())
                
                # Perform clustering
                print("Performing threshold-based clustering...")
                clusters = self.clustering_engine.threshold_clustering(
                    similarity_matrix, 
                    doc_ids, 
                    threshold=CLUSTERING.get("merge_threshold", 0.7)
                )
                
                print(f"✅ Created {len(clusters)} clusters:")
                
                for cluster_id, cluster in clusters.items():
                    doc_names = [Path(self.processed_documents[doc_id]).name 
                               for doc_id in cluster.documents]
                    print(f"  📁 {cluster_id}: {len(cluster.documents)} documents")
                    for name in doc_names[:3]:  # Show first 3
                        print(f"    - {name}")
                    if len(doc_names) > 3:
                        print(f"    ... and {len(doc_names) - 3} more")
                
            else:
                print("❌ Document vectors not available for clustering")
                
        except Exception as e:
            logger.error(f"Clustering error: {str(e)}")
            print(f"❌ Clustering error: {str(e)}")
    
    def demo_alert_system(self):
        """Demonstrate the alert system with a new document"""
        print("Testing alert system with existing documents...")
        
        try:
            # Take the first processed document and simulate it as "new"
            if self.processed_documents:
                test_doc_id = list(self.processed_documents.keys())[0]
                test_file_path = self.processed_documents[test_doc_id]
                
                # Remove it from the alert system's processed set
                self.alert_system.processed_documents.discard(test_doc_id)
                
                print(f"Processing '{Path(test_file_path).name}' as new document...")
                
                # Process as new document
                alert = self.alert_system.process_new_document(
                    document_id=f"new_{test_doc_id}",
                    file_path=test_file_path
                )
                
                if alert:
                    print(f"🚨 Alert generated!")
                    print(f"  Type: {alert.alert_type}")
                    print(f"  Severity: {alert.severity}")
                    print(f"  Similar documents found: {len(alert.similar_documents)}")
                    
                    for doc_id, similarity in alert.similar_documents[:3]:
                        print(f"    • {doc_id}: {similarity:.3f} similarity")
                else:
                    print("ℹ️  No alert generated (no similar documents found)")
                
                # Show alert statistics
                stats = self.alert_system.get_statistics()
                print(f"📊 Alert system stats:")
                print(f"  Total alerts: {stats['total_alerts']}")
                print(f"  Documents processed: {stats['total_documents_processed']}")
                
            else:
                print("⚠️  No documents available for alert testing")
                
        except Exception as e:
            logger.error(f"Alert system error: {str(e)}")
            print(f"❌ Alert system error: {str(e)}")
    
    def show_statistics(self):
        """Display comprehensive system statistics"""
        print("\n📊 System Statistics:")
        print("-" * 40)
        
        # Document processing stats
        print(f"📁 Documents processed: {len(self.processed_documents)}")
        
        # Index statistics
        try:
            index_stats = self.document_index.get_index_statistics()
            print(f"📚 Indexed documents: {index_stats.get('total_documents', 0)}")
            print(f"🔤 Unique terms: {index_stats.get('unique_terms', 0)}")
            print(f"💾 Index memory usage: {index_stats.get('memory_usage_mb', 0):.1f} MB")
            
            # Document type distribution
            type_dist = index_stats.get('doc_type_distribution', {})
            if type_dist:
                print("📄 Document types:")
                for doc_type, count in type_dist.items():
                    print(f"  {doc_type}: {count}")
                    
        except Exception as e:
            print(f"❌ Error getting index statistics: {str(e)}")
        
        # Alert system statistics
        try:
            alert_stats = self.alert_system.get_statistics()
            print(f"🚨 Total alerts: {alert_stats.get('total_alerts', 0)}")
            print(f"⏱️  Avg processing time: {alert_stats.get('avg_processing_time', 0):.3f}s")
            
        except Exception as e:
            print(f"❌ Error getting alert statistics: {str(e)}")
    
    def process_directory(self, directory_path: str, file_types: List[str] = None):
        """Process all documents in a directory"""
        print(f"Processing directory: {directory_path}")
        
        if file_types is None:
            file_types = [".pdf", ".txt", ".docx"]
        
        file_extensions = set(file_types)
        results = self.index_manager.index_directory(directory_path, file_extensions)
        
        success_count = sum(results.values())
        total_count = len(results)
        
        print(f"✅ Successfully processed {success_count}/{total_count} documents")
        
        # Save indices
        self.document_index.save_index()
        
        return results
    
    def search_documents(self, query: str, top_k: int = 10, method: str = "tfidf"):
        """Search for documents similar to query"""
        print(f"Searching for: '{query}' (top {top_k} results)")
        
        try:
            if method == "faiss":
                results = self.document_index.faiss_similarity_search(query, top_k)
            else:
                results = self.document_index.similarity_search(query, top_k)
            
            if results:
                print("📋 Search Results:")
                for i, (doc_id, similarity) in enumerate(results, 1):
                    metadata = self.document_index.get_document_metadata(doc_id)
                    file_name = Path(metadata.file_path).name if metadata else doc_id
                    print(f"  {i}. {file_name} (similarity: {similarity:.3f})")
            else:
                print("No results found")
                
            return results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            print(f"❌ Search error: {str(e)}")
            return []
    
    def cluster_documents(self, threshold: float = 0.7, algorithm: str = "threshold"):
        """Perform document clustering"""
        print(f"Clustering documents (threshold: {threshold}, algorithm: {algorithm})")
        
        try:
            # Build vectors if needed
            if self.document_index.document_vectors is None:
                print("Building document vectors...")
                self.document_index.build_vector_index()
            
            # Compute similarity matrix
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(self.document_index.document_vectors)
            
            # Get document IDs
            doc_ids = list(self.document_index.metadata.keys())
            
            if algorithm == "threshold":
                clusters = self.clustering_engine.threshold_clustering(
                    similarity_matrix, doc_ids, threshold
                )
            else:
                print(f"❌ Clustering algorithm '{algorithm}' not implemented")
                return None
            
            print(f"✅ Created {len(clusters)} clusters")
            
            # Save clustering results
            cluster_results = {}
            for cluster_id, cluster in clusters.items():
                cluster_results[cluster_id] = {
                    'documents': cluster.documents,
                    'size': len(cluster.documents),
                    'created_at': datetime.now().isoformat()
                }
            
            # Save to file
            output_file = CLUSTERS_DIR / f"clustering_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(cluster_results, f, indent=2)
            
            print(f"📁 Clustering results saved to: {output_file}")
            
            return clusters
            
        except Exception as e:
            logger.error(f"Clustering error: {str(e)}")
            print(f"❌ Clustering error: {str(e)}")
            return None
    
    def find_duplicates(self, threshold: float = 0.9):
        """Find duplicate documents"""
        print(f"Finding duplicates (similarity threshold: {threshold})")
        
        duplicates = self.index_manager.find_duplicates(threshold)
        
        if duplicates:
            print(f"🔍 Found {len(duplicates)} duplicate groups:")
            for i, group in enumerate(duplicates, 1):
                print(f"  Group {i}:")
                for doc_id in group:
                    metadata = self.document_index.get_document_metadata(doc_id)
                    file_name = Path(metadata.file_path).name if metadata else doc_id
                    print(f"    - {file_name}")
        else:
            print("✅ No duplicates found")
        
        return duplicates
    
    def export_results(self, output_dir: str = None):
        """Export all results and statistics"""
        if output_dir is None:
            output_dir = DATA_DIR / "exports" / datetime.now().strftime('%Y%m%d_%H%M%S')
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Exporting results to: {output_path}")
        
        # Export index report
        index_report_file = output_path / "index_report.json"
        self.index_manager.export_index_report(str(index_report_file))
        
        # Export alerts
        alerts_file = output_path / "alerts.json"
        self.alert_system.export_alerts(str(alerts_file), format="json")
        
        # Export alerts CSV
        alerts_csv_file = output_path / "alerts.csv"
        self.alert_system.export_alerts(str(alerts_csv_file), format="csv")
        
        print(f"✅ Results exported successfully")
        print(f"  - Index report: {index_report_file}")
        print(f"  - Alerts (JSON): {alerts_file}")
        print(f"  - Alerts (CSV): {alerts_csv_file}")


def create_argument_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Document Similarity Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s demo                                    # Run interactive demo
    %(prog)s process -d data/sample_pdfs            # Process directory
    %(prog)s search -q "machine learning" -k 5      # Search documents
    %(prog)s cluster -t 0.8                         # Cluster with threshold 0.8
    %(prog)s duplicates -t 0.95                     # Find duplicates
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run interactive demonstration')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process documents in directory')
    process_parser.add_argument('-d', '--directory', required=True, 
                               help='Directory containing documents to process')
    process_parser.add_argument('--types', nargs='+', default=['.pdf', '.txt', '.docx'],
                               help='File types to process (default: .pdf .txt .docx)')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for similar documents')
    search_parser.add_argument('-q', '--query', required=True, help='Search query')
    search_parser.add_argument('-k', '--top-k', type=int, default=10, 
                              help='Number of results to return (default: 10)')
    search_parser.add_argument('--method', choices=['tfidf', 'faiss'], default='tfidf',
                              help='Search method (default: tfidf)')
    
    # Cluster command
    cluster_parser = subparsers.add_parser('cluster', help='Perform document clustering')
    cluster_parser.add_argument('-t', '--threshold', type=float, default=0.7,
                               help='Similarity threshold for clustering (default: 0.7)')
    cluster_parser.add_argument('--algorithm', choices=['threshold'], default='threshold',
                               help='Clustering algorithm (default: threshold)')
    
    # Duplicates command
    duplicates_parser = subparsers.add_parser('duplicates', help='Find duplicate documents')
    duplicates_parser.add_argument('-t', '--threshold', type=float, default=0.9,
                                  help='Similarity threshold for duplicates (default: 0.9)')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export results and statistics')
    export_parser.add_argument('-o', '--output', help='Output directory')
    
    return parser


def main():
    """Main entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Initialize application
    try:
        app = DocumentSimilarityApp()
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        print(f"❌ Initialization failed: {str(e)}")
        sys.exit(1)
    
    # Execute command
    if args.command == 'demo' or args.command is None:
        app.run_demo()
        
    elif args.command == 'process':
        app.process_directory(args.directory, args.types)
        
    elif args.command == 'search':
        app.search_documents(args.query, args.top_k, args.method)
        
    elif args.command == 'cluster':
        app.cluster_documents(args.threshold, args.algorithm)
        
    elif args.command == 'duplicates':
        app.find_duplicates(args.threshold)
        
    elif args.command == 'export':
        app.export_results(args.output)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
