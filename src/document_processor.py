"""
Document processing utilities for text extraction and preprocessing
Supports multiple file formats including PDF, TXT, DOC, DOCX
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import mimetypes

# PDF processing
try:
    import PyPDF2
    import pdfplumber
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("PDF processing libraries not available")

# DOC/DOCX processing
try:
    from docx import Document
    DOC_AVAILABLE = True
except ImportError:
    DOC_AVAILABLE = False
    logging.warning("DOCX processing library not available")

from config.settings import *

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handles document text extraction and preprocessing
    Supports multiple file formats and extraction methods
    """
    
    def __init__(self, config: Dict = None):
        """Initialize document processor with configuration"""
        self.config = config or {}
        self.supported_extensions = SUPPORTED_EXTENSIONS
        self.text_config = TEXT_PROCESSING
        
        # Statistics
        self.processed_count = 0
        self.failed_count = 0
        self.processing_stats = {}
    
    def extract_text_from_pdf(self, pdf_path: str, method: str = "pdfplumber") -> str:
        """
        Extract text from PDF using specified method
        
        Args:
            pdf_path: Path to PDF file
            method: Extraction method ("pdfplumber", "pymupdf", "pypdf2")
            
        Returns:
            Extracted text
        """
        if not PDF_AVAILABLE:
            raise ImportError("PDF processing libraries not installed")
        
        text = ""
        
        try:
            if method == "pdfplumber":
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            
            elif method == "pymupdf":
                doc = fitz.open(pdf_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text += page.get_text() + "\n"
                doc.close()
            
            elif method == "pypdf2":
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            
            else:
                raise ValueError(f"Unsupported PDF extraction method: {method}")
        
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {str(e)}")
            return ""
        
        return text.strip()
    
    def extract_text_from_docx(self, docx_path: str) -> str:
        """
        Extract text from DOCX file
        
        Args:
            docx_path: Path to DOCX file
            
        Returns:
            Extracted text
        """
        if not DOC_AVAILABLE:
            raise ImportError("DOCX processing library not installed")
        
        try:
            doc = Document(docx_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text.strip()
        
        except Exception as e:
            logger.error(f"Failed to extract text from {docx_path}: {str(e)}")
            return ""
    
    def extract_text_from_txt(self, txt_path: str, encoding: str = "utf-8") -> str:
        """
        Extract text from TXT file
        
        Args:
            txt_path: Path to TXT file
            encoding: Text encoding
            
        Returns:
            Extracted text
        """
        try:
            with open(txt_path, 'r', encoding=encoding) as file:
                return file.read().strip()
        
        except UnicodeDecodeError:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
            for enc in encodings:
                try:
                    with open(txt_path, 'r', encoding=enc) as file:
                        logger.warning(f"Used encoding {enc} for {txt_path}")
                        return file.read().strip()
                except UnicodeDecodeError:
                    continue
            
            logger.error(f"Failed to decode {txt_path} with any encoding")
            return ""
        
        except Exception as e:
            logger.error(f"Failed to read {txt_path}: {str(e)}")
            return ""
    
    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text from file based on extension
        
        Args:
            file_path: Path to file
            
        Returns:
            Extracted text
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension not in self.supported_extensions:
            logger.warning(f"Unsupported file extension: {extension}")
            return ""
        
        try:
            if extension == ".pdf":
                return self.extract_text_from_pdf(str(path))
            elif extension == ".docx":
                return self.extract_text_from_docx(str(path))
            elif extension == ".txt":
                return self.extract_text_from_txt(str(path))
            elif extension in [".doc", ".rtf"]:
                logger.warning(f"Limited support for {extension} files")
                return self.extract_text_from_txt(str(path))
            else:
                return ""
        
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {str(e)}")
            self.failed_count += 1
            return ""
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
        
        # Remove extra spaces around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s+', r'\1 ', text)
        
        # Normalize case (optional)
        # text = text.lower()
        
        return text.strip()
    
    def validate_document(self, text: str) -> bool:
        """
        Validate if document meets minimum requirements
        
        Args:
            text: Document text
            
        Returns:
            True if valid, False otherwise
        """
        if not text:
            return False
        
        # Check minimum length
        if len(text) < self.text_config["min_doc_length"]:
            return False
        
        # Check maximum length
        if len(text) > self.text_config["max_doc_length"]:
            logger.warning(f"Document too long ({len(text)} chars), truncating")
            return True
        
        # Check for meaningful content (not just whitespace/punctuation)
        word_count = len(re.findall(r'\w+', text))
        if word_count < 10:  # Minimum 10 words
            return False
        
        return True
    
    def process_single_document(self, file_path: str) -> Tuple[str, str]:
        """
        Process a single document and return document ID and text
        
        Args:
            file_path: Path to document file
            
        Returns:
            Tuple of (document_id, processed_text)
        """
        path = Path(file_path)
        document_id = path.stem  # Use filename without extension as ID
        
        # Extract text
        raw_text = self.extract_text_from_file(file_path)
        
        if not raw_text:
            logger.warning(f"No text extracted from {file_path}")
            return document_id, ""
        
        # Clean text
        cleaned_text = self.clean_text(raw_text)
        
        # Validate document
        if not self.validate_document(cleaned_text):
            logger.warning(f"Document validation failed for {file_path}")
            return document_id, ""
        
        # Truncate if too long
        if len(cleaned_text) > self.text_config["max_doc_length"]:
            cleaned_text = cleaned_text[:self.text_config["max_doc_length"]]
        
        self.processed_count += 1
        
        # Update statistics
        self.processing_stats[document_id] = {
            "file_path": str(file_path),
            "file_size": path.stat().st_size,
            "text_length": len(cleaned_text),
            "word_count": len(re.findall(r'\w+', cleaned_text))
        }
        
        return document_id, cleaned_text
    
    def process_directory(self, directory_path: str, 
                         recursive: bool = True) -> Dict[str, str]:
        """
        Process all documents in a directory
        
        Args:
            directory_path: Path to directory
            recursive: Whether to process subdirectories
            
        Returns:
            Dictionary of {document_id: document_text}
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        documents = {}
        
        # Get all files with supported extensions
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                logger.info(f"Processing {file_path}")
                
                doc_id, text = self.process_single_document(str(file_path))
                
                if text:  # Only add if text was successfully extracted
                    documents[doc_id] = text
        
        logger.info(f"Processed {len(documents)} documents from {directory_path}")
        logger.info(f"Success: {self.processed_count}, Failed: {self.failed_count}")
        
        return documents
    
    def process_file_list(self, file_paths: List[str]) -> Dict[str, str]:
        """
        Process a list of document files
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Dictionary of {document_id: document_text}
        """
        documents = {}
        
        for file_path in file_paths:
            logger.info(f"Processing {file_path}")
            
            doc_id, text = self.process_single_document(file_path)
            
            if text:  # Only add if text was successfully extracted
                documents[doc_id] = text
        
        logger.info(f"Processed {len(documents)} documents from file list")
        logger.info(f"Success: {self.processed_count}, Failed: {self.failed_count}")
        
        return documents
    
    def load_processed_documents(self, processed_dir: str) -> Dict[str, str]:
        """
        Load already processed documents from text files
        
        Args:
            processed_dir: Directory containing processed text files
            
        Returns:
            Dictionary of {document_id: document_text}
        """
        processed_path = Path(processed_dir)
        
        if not processed_path.exists():
            logger.warning(f"Processed documents directory not found: {processed_dir}")
            return {}
        
        documents = {}
        
        # Load all .txt files from processed directory
        for txt_file in processed_path.glob("*.txt"):
            doc_id = txt_file.stem
            text = self.extract_text_from_txt(str(txt_file))
            
            if text and self.validate_document(text):
                documents[doc_id] = text
                self.processed_count += 1
        
        logger.info(f"Loaded {len(documents)} processed documents")
        return documents
    
    def save_processed_documents(self, documents: Dict[str, str], 
                               output_dir: str):
        """
        Save processed documents to text files
        
        Args:
            documents: Dictionary of {document_id: document_text}
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for doc_id, text in documents.items():
            output_file = output_path / f"{doc_id}.txt"
            
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                    
            except Exception as e:
                logger.error(f"Failed to save {doc_id}: {str(e)}")
        
        logger.info(f"Saved {len(documents)} processed documents to {output_dir}")
    
    def get_processing_stats(self) -> Dict:
        """
        Get processing statistics
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            "processed_count": self.processed_count,
            "failed_count": self.failed_count,
            "success_rate": self.processed_count / (self.processed_count + self.failed_count) if (self.processed_count + self.failed_count) > 0 else 0,
            "document_stats": self.processing_stats
        }
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.processed_count = 0
        self.failed_count = 0
        self.processing_stats = {}
    
    def detect_file_encoding(self, file_path: str) -> str:
        """
        Detect file encoding using various methods
        
        Args:
            file_path: Path to file
            
        Returns:
            Detected encoding
        """
        try:
            import chardet
            
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                return result.get('encoding', 'utf-8')
        
        except ImportError:
            logger.warning("chardet not available, using utf-8 as default")
            return 'utf-8'
        
        except Exception as e:
            logger.error(f"Encoding detection failed: {str(e)}")
            return 'utf-8'