"""
Token-aware text chunking for AAOIFI content.
"""

import tiktoken
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def get_token_count(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Get token count for text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback estimation
        return len(text.split()) * 1.3

def chunk_text(text: str, max_tokens: int = 1000, overlap: int = 100) -> List[Dict]:
    """Chunk text into token-aware segments with overlap."""
    chunks = []
    sentences = text.split('.')

    current_chunk = ""
    chunk_id = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        test_chunk = current_chunk + ". " + sentence if current_chunk else sentence

        if get_token_count(test_chunk) > max_tokens and current_chunk:
            # Save current chunk
            chunks.append({
                "id": chunk_id,
                "text": current_chunk.strip(),
                "token_count": get_token_count(current_chunk),
                "start_sentence": len(chunks) * (max_tokens // 100),  # Approximate
                "overlap_tokens": overlap if chunk_id > 0 else 0
            })

            # Start new chunk with overlap
            overlap_text = ". ".join(current_chunk.split(". ")[-2:]) if "." in current_chunk else ""
            current_chunk = overlap_text + ". " + sentence if overlap_text else sentence
            chunk_id += 1
        else:
            current_chunk = test_chunk

    # Add final chunk
    if current_chunk:
        chunks.append({
            "id": chunk_id,
            "text": current_chunk.strip(),
            "token_count": get_token_count(current_chunk),
            "start_sentence": len(chunks) * (max_tokens // 100),
            "overlap_tokens": overlap if chunk_id > 0 else 0
        })

    logger.info(f"Created {len(chunks)} chunks from text ({get_token_count(text)} total tokens)")
    return chunks