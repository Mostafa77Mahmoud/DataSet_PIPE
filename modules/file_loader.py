import os
import logging
from typing import Tuple
from docx import Document

class FileLoader:
    """Class to handle local file loading as alternative to Google Docs."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def _extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        self.logger.info(f"Extracting text from DOCX file: {file_path}")
        try:
            doc = Document(file_path)
            text_content = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text.strip())
            
            # Join paragraphs with double newlines to preserve structure
            text = '\n\n'.join(text_content)
            self.logger.info(f"Extracted {len(text_content)} paragraphs from DOCX")
            return text
            
        except Exception as e:
            self.logger.error(f"Failed to extract text from DOCX: {e}")
            raise

    def _load_file(self, file_path: str) -> str:
        """Load text from file (supports .txt, .docx)."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.docx':
            return self._extract_text_from_docx(file_path)
        elif file_ext in ['.txt', '.text']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # Try to read as text file for unknown extensions
            self.logger.warning(f"Unknown file extension {file_ext}, attempting to read as text")
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

    def load_local_files(self, ar_file_path: str, en_file_path: str) -> Tuple[str, str]:
        """Load Arabic and English text from local files (supports .txt, .docx)."""
        self.logger.info("Loading documents from local files")
        
        # Load Arabic file
        self.logger.info(f"Loading Arabic file: {ar_file_path}")
        ar_text = self._load_file(ar_file_path)
        
        # Load English file  
        self.logger.info(f"Loading English file: {en_file_path}")
        en_text = self._load_file(en_file_path)
        
        self.logger.info(f"Arabic text length: {len(ar_text)} characters")
        self.logger.info(f"English text length: {len(en_text)} characters")
        
        return ar_text, en_text