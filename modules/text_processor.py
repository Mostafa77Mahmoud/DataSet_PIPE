import re
import logging
from typing import List, Tuple, Dict, Any
from modules.utils import save_intermediate_file

class TextProcessor:
    """Class to handle text preprocessing and chunking."""
    
    def __init__(self, chunk_size: int = 1500):
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
    
    def preprocess_text(self, text: str, language: str) -> str:
        """Preprocess text by cleaning and normalizing."""
        self.logger.info(f"Preprocessing {language} text")
        
        # Remove headers, footers, and metadata
        text = self._remove_metadata(text)
        
        # Normalize punctuation based on language
        if language == "arabic":
            text = self._normalize_arabic_punctuation(text)
        else:
            text = self._normalize_english_punctuation(text)
        
        # Remove extra spaces and empty lines
        text = self._clean_whitespace(text)
        
        self.logger.info(f"Preprocessed {language} text length: {len(text)} characters")
        return text
    
    def _remove_metadata(self, text: str) -> str:
        """Remove headers, footers, TOC, page numbers, and other metadata."""
        # Patterns to remove
        patterns_to_remove = [
            r'^.*?Table of Contents.*?(?=\n\n|\n[A-Z])',  # TOC
            r'^.*?Contents.*?(?=\n\n|\n[A-Z])',  # Contents
            r'Page \d+.*?\n',  # Page numbers
            r'^\d+\s*$',  # Standalone page numbers
            r'^.*?Header.*?\n',  # Headers
            r'^.*?Footer.*?\n',  # Footers
            r'^\s*\d+\s*\n',  # Standalone numbers on lines
            r'Copyright.*?\n',  # Copyright notices
            r'All rights reserved.*?\n',  # Rights notices
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        return text
    
    def _normalize_arabic_punctuation(self, text: str) -> str:
        """Normalize Arabic punctuation and text."""
        # Replace various forms of Arabic punctuation
        replacements = {
            '؟': '؟',  # Arabic question mark
            '؛': '؛',  # Arabic semicolon
            '،': '،',  # Arabic comma
            'ي': 'ي',  # Normalize Arabic ya
            'ك': 'ك',  # Normalize Arabic kaf
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Fix spacing around Arabic punctuation
        text = re.sub(r'\s+([؟؛،])', r'\1', text)
        text = re.sub(r'([؟؛،])(?!\s)', r'\1 ', text)
        
        return text
    
    def _normalize_english_punctuation(self, text: str) -> str:
        """Normalize English punctuation."""
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.!?;,:])', r'\1', text)
        text = re.sub(r'([.!?])(?!\s)', r'\1 ', text)
        text = re.sub(r'([,;:])(?!\s)', r'\1 ', text)
        
        # Fix quotation marks
        text = re.sub(r'"([^"]*)"', r'"\1"', text)
        text = re.sub(r"'([^']*)'", r"'\1'", text)
        
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean up whitespace and empty lines."""
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove multiple line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Remove completely empty lines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def chunk_text(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Split text into chunks while preserving paragraph boundaries."""
        self.logger.info(f"Chunking {language} text into ~{self.chunk_size} word chunks")
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        current_word_count = 0
        chunk_id = 0
        
        for paragraph in paragraphs:
            paragraph_words = len(paragraph.split())
            
            # If adding this paragraph would exceed chunk size and current chunk is not empty
            if current_word_count + paragraph_words > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'id': chunk_id,
                    'text': current_chunk.strip(),
                    'word_count': current_word_count,
                    'language': language
                })
                
                # Start new chunk
                chunk_id += 1
                current_chunk = paragraph
                current_word_count = paragraph_words
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_word_count += paragraph_words
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append({
                'id': chunk_id,
                'text': current_chunk.strip(),
                'word_count': current_word_count,
                'language': language
            })
        
        self.logger.info(f"Created {len(chunks)} chunks for {language} text")
        
        # Save chunks for intermediate review
        save_intermediate_file(chunks, f"{language}_chunks.json", "json")
        
        return chunks
    
    def align_chunks(self, ar_chunks: List[Dict], en_chunks: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """Align Arabic and English chunks by position and content similarity."""
        self.logger.info("Aligning Arabic and English chunks")
        
        aligned_chunks = []
        min_chunks = min(len(ar_chunks), len(en_chunks))
        
        for i in range(min_chunks):
            aligned_chunks.append((ar_chunks[i], en_chunks[i]))
        
        # Handle mismatched chunk counts
        if len(ar_chunks) != len(en_chunks):
            self.logger.warning(f"Chunk count mismatch: Arabic={len(ar_chunks)}, English={len(en_chunks)}")
            self.logger.warning(f"Using {min_chunks} aligned chunks")
        
        self.logger.info(f"Created {len(aligned_chunks)} aligned chunk pairs")
        return aligned_chunks
    
    def process_documents(self, ar_text: str, en_text: str) -> List[Tuple[Dict, Dict]]:
        """Complete preprocessing and chunking pipeline."""
        self.logger.info("Starting text processing pipeline")
        
        # Preprocess texts
        ar_clean = self.preprocess_text(ar_text, "arabic")
        en_clean = self.preprocess_text(en_text, "english")
        
        # Save cleaned texts
        save_intermediate_file(ar_clean, "arabic_cleaned.txt", "text")
        save_intermediate_file(en_clean, "english_cleaned.txt", "text")
        
        # Chunk texts
        ar_chunks = self.chunk_text(ar_clean, "arabic")
        en_chunks = self.chunk_text(en_clean, "english")
        
        # Align chunks
        aligned_chunks = self.align_chunks(ar_chunks, en_chunks)
        
        self.logger.info("Text processing pipeline completed")
        return aligned_chunks
