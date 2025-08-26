
import logging
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import difflib

@dataclass
class AlignmentResult:
    """Result of chunk alignment"""
    ar_chunk_id: int
    en_chunk_id: int
    confidence: float
    method: str  # 'heading', 'position', 'semantic', 'unmatched'

class AlignmentEngine:
    """Advanced alignment engine for Arabic-English content"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.embedding_model = None
        self._init_embedding_model()
    
    def _init_embedding_model(self):
        """Initialize multilingual embedding model for semantic alignment"""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.logger.info("Loaded multilingual embedding model")
        except ImportError:
            self.logger.warning("sentence-transformers not available, using text-based alignment only")
        except Exception as e:
            self.logger.warning(f"Could not load embedding model: {e}")
    
    def normalize_heading(self, text: str) -> str:
        """Normalize heading text for comparison"""
        # Remove numbers, punctuation, extra spaces
        text = re.sub(r'[\d\.\)\-\:]+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()
    
    def extract_headings(self, chunks: List[Dict]) -> List[Dict]:
        """Extract and normalize headings from chunks"""
        headings = []
        
        for chunk in chunks:
            text = chunk['text']
            lines = text.split('\n')
            
            for i, line in enumerate(lines[:5]):  # Check first 5 lines
                line = line.strip()
                
                # Heading detection heuristics
                is_heading = (
                    len(line) < 200 and  # Not too long
                    len(line) > 5 and    # Not too short
                    (line.isupper() or   # All caps
                     re.match(r'^\d+[\.\)]\s+', line) or  # Numbered
                     re.match(r'^[IVX]+[\.\)]\s+', line) or  # Roman numerals
                     line.endswith(':') or  # Ends with colon
                     (i == 0 and len(lines) > 1))  # First line
                )
                
                if is_heading:
                    headings.append({
                        'chunk_id': chunk['chunk_id'],
                        'original': line,
                        'normalized': self.normalize_heading(line),
                        'position': i,
                        'language': chunk['language']
                    })
                    break  # Only take first heading per chunk
        
        return headings
    
    def align_by_headings(self, ar_chunks: List[Dict], en_chunks: List[Dict]) -> List[AlignmentResult]:
        """Align chunks based on heading similarity"""
        self.logger.info("Attempting heading-based alignment")
        
        ar_headings = self.extract_headings(ar_chunks)
        en_headings = self.extract_headings(en_chunks)
        
        alignments = []
        used_en_ids = set()
        
        for ar_heading in ar_headings:
            best_match = None
            best_score = 0.0
            
            for en_heading in en_headings:
                if en_heading['chunk_id'] in used_en_ids:
                    continue
                
                # Calculate similarity using difflib
                similarity = difflib.SequenceMatcher(
                    None, 
                    ar_heading['normalized'], 
                    en_heading['normalized']
                ).ratio()
                
                if similarity > best_score and similarity > 0.3:  # Minimum threshold
                    best_score = similarity
                    best_match = en_heading
            
            if best_match:
                alignments.append(AlignmentResult(
                    ar_chunk_id=ar_heading['chunk_id'],
                    en_chunk_id=best_match['chunk_id'],
                    confidence=best_score,
                    method='heading'
                ))
                used_en_ids.add(best_match['chunk_id'])
        
        self.logger.info(f"Heading-based alignment found {len(alignments)} matches")
        return alignments
    
    def align_by_position(self, ar_chunks: List[Dict], en_chunks: List[Dict], 
                         existing_alignments: List[AlignmentResult]) -> List[AlignmentResult]:
        """Align remaining chunks by relative position"""
        self.logger.info("Attempting position-based alignment")
        
        # Get already aligned chunk IDs
        aligned_ar_ids = {a.ar_chunk_id for a in existing_alignments}
        aligned_en_ids = {a.en_chunk_id for a in existing_alignments}
        
        # Get unaligned chunks
        unaligned_ar = [c for c in ar_chunks if c['chunk_id'] not in aligned_ar_ids]
        unaligned_en = [c for c in en_chunks if c['chunk_id'] not in aligned_en_ids]
        
        # Sort by chunk_id (position in document)
        unaligned_ar.sort(key=lambda x: x['chunk_id'])
        unaligned_en.sort(key=lambda x: x['chunk_id'])
        
        new_alignments = []
        min_chunks = min(len(unaligned_ar), len(unaligned_en))
        
        for i in range(min_chunks):
            ar_chunk = unaligned_ar[i]
            en_chunk = unaligned_en[i]
            
            # Calculate confidence based on relative position
            ar_pos = ar_chunk['chunk_id'] / len(ar_chunks)
            en_pos = en_chunk['chunk_id'] / len(en_chunks)
            position_diff = abs(ar_pos - en_pos)
            confidence = max(0.1, 1.0 - position_diff * 2)  # Higher confidence for closer positions
            
            new_alignments.append(AlignmentResult(
                ar_chunk_id=ar_chunk['chunk_id'],
                en_chunk_id=en_chunk['chunk_id'],
                confidence=confidence,
                method='position'
            ))
        
        self.logger.info(f"Position-based alignment found {len(new_alignments)} matches")
        return existing_alignments + new_alignments
    
    def align_by_semantics(self, ar_chunks: List[Dict], en_chunks: List[Dict], 
                          existing_alignments: List[AlignmentResult]) -> List[AlignmentResult]:
        """Align chunks using semantic similarity (if embedding model available)"""
        if not self.embedding_model:
            self.logger.info("Skipping semantic alignment - no embedding model available")
            return existing_alignments
        
        self.logger.info("Attempting semantic alignment")
        
        # Get already aligned chunk IDs
        aligned_ar_ids = {a.ar_chunk_id for a in existing_alignments}
        aligned_en_ids = {a.en_chunk_id for a in existing_alignments}
        
        # Get unaligned chunks
        unaligned_ar = [c for c in ar_chunks if c['chunk_id'] not in aligned_ar_ids]
        unaligned_en = [c for c in en_chunks if c['chunk_id'] not in aligned_en_ids]
        
        if not unaligned_ar or not unaligned_en:
            return existing_alignments
        
        try:
            # Generate embeddings
            ar_texts = [chunk['text'][:1000] for chunk in unaligned_ar]  # Limit text length
            en_texts = [chunk['text'][:1000] for chunk in unaligned_en]
            
            ar_embeddings = self.embedding_model.encode(ar_texts)
            en_embeddings = self.embedding_model.encode(en_texts)
            
            # Find best matches using cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(ar_embeddings, en_embeddings)
            
            new_alignments = []
            used_en_indices = set()
            
            for ar_idx, ar_chunk in enumerate(unaligned_ar):
                best_en_idx = None
                best_score = 0.0
                
                for en_idx in range(len(unaligned_en)):
                    if en_idx in used_en_indices:
                        continue
                    
                    score = similarity_matrix[ar_idx][en_idx]
                    if score > best_score and score > 0.5:  # Minimum threshold
                        best_score = score
                        best_en_idx = en_idx
                
                if best_en_idx is not None:
                    en_chunk = unaligned_en[best_en_idx]
                    new_alignments.append(AlignmentResult(
                        ar_chunk_id=ar_chunk['chunk_id'],
                        en_chunk_id=en_chunk['chunk_id'],
                        confidence=best_score,
                        method='semantic'
                    ))
                    used_en_indices.add(best_en_idx)
            
            self.logger.info(f"Semantic alignment found {len(new_alignments)} matches")
            return existing_alignments + new_alignments
            
        except Exception as e:
            self.logger.error(f"Semantic alignment failed: {e}")
            return existing_alignments
    
    def mark_unmatched(self, ar_chunks: List[Dict], en_chunks: List[Dict], 
                      alignments: List[AlignmentResult]) -> List[AlignmentResult]:
        """Mark remaining chunks as unmatched"""
        aligned_ar_ids = {a.ar_chunk_id for a in alignments}
        aligned_en_ids = {a.en_chunk_id for a in alignments}
        
        unmatched_alignments = []
        
        # Mark unmatched Arabic chunks
        for chunk in ar_chunks:
            if chunk['chunk_id'] not in aligned_ar_ids:
                unmatched_alignments.append(AlignmentResult(
                    ar_chunk_id=chunk['chunk_id'],
                    en_chunk_id=-1,  # No match
                    confidence=0.0,
                    method='unmatched'
                ))
        
        # Mark unmatched English chunks
        for chunk in en_chunks:
            if chunk['chunk_id'] not in aligned_en_ids:
                unmatched_alignments.append(AlignmentResult(
                    ar_chunk_id=-1,  # No match
                    en_chunk_id=chunk['chunk_id'],
                    confidence=0.0,
                    method='unmatched'
                ))
        
        if unmatched_alignments:
            self.logger.warning(f"Found {len(unmatched_alignments)} unmatched chunks")
        
        return alignments + unmatched_alignments
    
    def align_chunks(self, ar_chunks: List[Dict], en_chunks: List[Dict]) -> List[AlignmentResult]:
        """Complete alignment pipeline"""
        self.logger.info(f"Aligning {len(ar_chunks)} Arabic chunks with {len(en_chunks)} English chunks")
        
        # Step 1: Heading-based alignment
        alignments = self.align_by_headings(ar_chunks, en_chunks)
        
        # Step 2: Position-based alignment for remaining chunks
        alignments = self.align_by_position(ar_chunks, en_chunks, alignments)
        
        # Step 3: Semantic alignment (if available)
        alignments = self.align_by_semantics(ar_chunks, en_chunks, alignments)
        
        # Step 4: Mark unmatched chunks
        alignments = self.mark_unmatched(ar_chunks, en_chunks, alignments)
        
        # Generate alignment statistics
        method_counts = {}
        confidence_sum = 0
        matched_count = 0
        
        for alignment in alignments:
            method_counts[alignment.method] = method_counts.get(alignment.method, 0) + 1
            if alignment.method != 'unmatched':
                confidence_sum += alignment.confidence
                matched_count += 1
        
        avg_confidence = confidence_sum / matched_count if matched_count > 0 else 0
        
        self.logger.info(f"Alignment complete: {method_counts}")
        self.logger.info(f"Average confidence: {avg_confidence:.2f}")
        
        return alignments
    
    def create_aligned_chunk_pairs(self, ar_chunks: List[Dict], en_chunks: List[Dict], 
                                  alignments: List[AlignmentResult]) -> List[Tuple[Dict, Dict, AlignmentResult]]:
        """Create pairs of aligned chunks with metadata"""
        ar_chunk_map = {c['chunk_id']: c for c in ar_chunks}
        en_chunk_map = {c['chunk_id']: c for c in en_chunks}
        
        aligned_pairs = []
        
        for alignment in alignments:
            if alignment.method == 'unmatched':
                continue
            
            ar_chunk = ar_chunk_map.get(alignment.ar_chunk_id)
            en_chunk = en_chunk_map.get(alignment.en_chunk_id)
            
            if ar_chunk and en_chunk:
                aligned_pairs.append((ar_chunk, en_chunk, alignment))
        
        self.logger.info(f"Created {len(aligned_pairs)} aligned chunk pairs")
        return aligned_pairs
