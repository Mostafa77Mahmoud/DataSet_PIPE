
import tiktoken
import logging
import math
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json
import os

@dataclass
class TokenConfig:
    """Configuration for token management"""
    max_tokens_per_request: int = 250000
    safety_margin: float = 0.96
    expected_output_tokens: int = 2048
    per_chunk_expected_output: int = 512
    chunk_overlap_sentences: int = 2
    max_paragraph_tokens: int = 15000
    
    @property
    def effective_limit(self) -> int:
        return int(self.max_tokens_per_request * self.safety_margin)

@dataclass 
class ChunkMetadata:
    """Metadata for each chunk"""
    chunk_id: int
    language: str
    start_paragraph_index: int
    end_paragraph_index: int
    token_count: int
    batch_id: Optional[int] = None
    file_offset: int = 0
    
@dataclass
class BatchInfo:
    """Information about a batch of chunks"""
    batch_id: int
    chunk_ids: List[int]
    total_tokens: int
    prompt_tokens: int
    expected_output_tokens: int
    language: str

class TokenManager:
    """Manages tokenization and chunking with token awareness"""
    
    def __init__(self, config: TokenConfig = None):
        self.config = config or TokenConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize tokenizer - using cl100k_base which is compatible with GPT models
        # For Gemini, this provides a reasonable approximation
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            self.logger.warning(f"Could not load tiktoken, falling back to word estimation: {e}")
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tokenizer or estimation"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                self.logger.warning(f"Tokenizer failed, using estimation: {e}")
        
        # Fallback estimation: roughly 0.75 tokens per word for mixed text
        words = len(text.split())
        return int(words * 0.75)
    
    def split_into_paragraph_units(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Split text into paragraph units preserving structure"""
        paragraphs = []
        
        # Split by double newlines and clean
        raw_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        for i, paragraph in enumerate(raw_paragraphs):
            # Detect if this is a heading (simple heuristic)
            is_heading = (
                len(paragraph.split('\n')) == 1 and  # Single line
                len(paragraph) < 200 and  # Short
                (paragraph.isupper() or  # All caps
                 any(char.isdigit() for char in paragraph[:10]) or  # Starts with numbers
                 paragraph.endswith(':'))  # Ends with colon
            )
            
            token_count = self.count_tokens(paragraph)
            
            paragraphs.append({
                'index': i,
                'text': paragraph,
                'token_count': token_count,
                'is_heading': is_heading,
                'language': language
            })
        
        self.logger.info(f"Split {language} text into {len(paragraphs)} paragraph units")
        return paragraphs
    
    def create_chunks_with_overlap(self, paragraph_units: List[Dict], language: str) -> List[Dict]:
        """Create chunks from paragraph units with token awareness and overlap"""
        chunks = []
        chunk_id = 0
        
        # Target tokens per chunk (will be adjusted based on batching)
        target_tokens_per_chunk = min(self.config.max_paragraph_tokens, 15000)
        
        i = 0
        while i < len(paragraph_units):
            chunk_paragraphs = []
            current_tokens = 0
            start_index = i
            
            # Greedy packing until we reach target or can't fit next paragraph
            while i < len(paragraph_units):
                pu = paragraph_units[i]
                
                # Check if adding this paragraph would exceed limit
                if current_tokens + pu['token_count'] > target_tokens_per_chunk and chunk_paragraphs:
                    break
                
                chunk_paragraphs.append(pu)
                current_tokens += pu['token_count']
                i += 1
                
                # If this is a heading, consider it a natural break point
                if pu['is_heading'] and current_tokens > target_tokens_per_chunk * 0.5:
                    break
            
            # Create overlap with previous chunk
            if chunk_id > 0 and self.config.chunk_overlap_sentences > 0:
                # Add last few sentences from previous chunk as overlap
                overlap_text = self._get_overlap_text(chunks[-1]['text'], self.config.chunk_overlap_sentences)
                if overlap_text:
                    chunk_text = overlap_text + "\n\n" + "\n\n".join([p['text'] for p in chunk_paragraphs])
                    current_tokens += self.count_tokens(overlap_text)
                else:
                    chunk_text = "\n\n".join([p['text'] for p in chunk_paragraphs])
            else:
                chunk_text = "\n\n".join([p['text'] for p in chunk_paragraphs])
            
            if chunk_text.strip():
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'token_count': current_tokens,
                    'language': language,
                    'start_paragraph_index': start_index,
                    'end_paragraph_index': i - 1,
                    'paragraph_count': len(chunk_paragraphs)
                })
                chunk_id += 1
        
        self.logger.info(f"Created {len(chunks)} chunks for {language}, avg tokens: {sum(c['token_count'] for c in chunks) / len(chunks) if chunks else 0:.0f}")
        return chunks
    
    def _get_overlap_text(self, text: str, num_sentences: int) -> str:
        """Get last N sentences from text for overlap"""
        import re
        
        # Split into sentences (simple approach)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) >= num_sentences:
            overlap_sentences = sentences[-num_sentences:]
            return '. '.join(overlap_sentences) + '.'
        
        return ''
    
    def create_batches(self, chunks: List[Dict], language: str, prompt_template: str) -> List[BatchInfo]:
        """Create batches of chunks to maximize token utilization"""
        batches = []
        batch_id = 0
        
        # Estimate base prompt tokens
        base_prompt_tokens = self.count_tokens(prompt_template)
        
        i = 0
        while i < len(chunks):
            batch_chunks = []
            batch_tokens = base_prompt_tokens
            expected_output = 0
            
            # Pack chunks into batch until we hit effective limit
            while i < len(chunks):
                chunk = chunks[i]
                chunk_tokens = chunk['token_count']
                chunk_output_tokens = self.config.per_chunk_expected_output
                
                # Check if adding this chunk would exceed effective limit
                total_with_chunk = batch_tokens + chunk_tokens + expected_output + chunk_output_tokens
                
                if total_with_chunk > self.config.effective_limit and batch_chunks:
                    break
                
                batch_chunks.append(chunk['chunk_id'])
                batch_tokens += chunk_tokens
                expected_output += chunk_output_tokens
                i += 1
            
            if batch_chunks:
                batches.append(BatchInfo(
                    batch_id=batch_id,
                    chunk_ids=batch_chunks,
                    total_tokens=batch_tokens + expected_output,
                    prompt_tokens=batch_tokens,
                    expected_output_tokens=expected_output,
                    language=language
                ))
                
                # Update chunks with batch_id
                for chunk_id in batch_chunks:
                    for chunk in chunks:
                        if chunk['chunk_id'] == chunk_id:
                            chunk['batch_id'] = batch_id
                
                batch_id += 1
        
        utilization = sum(b.total_tokens for b in batches) / (len(batches) * self.config.effective_limit) if batches else 0
        self.logger.info(f"Created {len(batches)} batches for {language}, avg utilization: {utilization:.1%}")
        
        return batches
    
    def save_chunks_manifest(self, ar_chunks: List[Dict], en_chunks: List[Dict], 
                           ar_batches: List[BatchInfo], en_batches: List[BatchInfo]):
        """Save chunks and batches manifest for inspection"""
        os.makedirs("intermediate", exist_ok=True)
        
        manifest = {
            'config': {
                'max_tokens_per_request': self.config.max_tokens_per_request,
                'effective_limit': self.config.effective_limit,
                'safety_margin': self.config.safety_margin,
                'per_chunk_expected_output': self.config.per_chunk_expected_output
            },
            'arabic': {
                'chunks': ar_chunks,
                'batches': [self._batch_to_dict(b) for b in ar_batches],
                'total_chunks': len(ar_chunks),
                'total_batches': len(ar_batches),
                'total_tokens': sum(c['token_count'] for c in ar_chunks)
            },
            'english': {
                'chunks': en_chunks,
                'batches': [self._batch_to_dict(b) for b in en_batches],
                'total_chunks': len(en_chunks),
                'total_batches': len(en_batches),
                'total_tokens': sum(c['token_count'] for c in en_chunks)
            }
        }
        
        with open("intermediate/chunks_manifest.json", 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        self.logger.info("Saved chunks manifest to intermediate/chunks_manifest.json")
    
    def _batch_to_dict(self, batch: BatchInfo) -> dict:
        """Convert BatchInfo to dictionary for JSON serialization"""
        return {
            'batch_id': batch.batch_id,
            'chunk_ids': batch.chunk_ids,
            'total_tokens': batch.total_tokens,
            'prompt_tokens': batch.prompt_tokens,
            'expected_output_tokens': batch.expected_output_tokens,
            'language': batch.language,
            'utilization': batch.total_tokens / self.config.effective_limit
        }
