import json
import logging
from typing import List, Dict, Set
from collections import defaultdict
from modules.utils import save_intermediate_file

class DataMerger:
    """Class to merge and align bilingual Q&A data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def group_qa_by_chunk(self, qa_pairs: List[Dict]) -> Dict[int, List[Dict]]:
        """Group Q&A pairs by chunk ID."""
        grouped = defaultdict(list)
        for qa in qa_pairs:
            chunk_id = qa.get('chunk_id', 0)
            grouped[chunk_id].append(qa)
        return dict(grouped)
    
    def deduplicate_qa_pairs(self, qa_pairs: List[Dict], language: str) -> List[Dict]:
        """Remove duplicate Q&A pairs within a language."""
        self.logger.info(f"Deduplicating {language} Q&A pairs")
        
        seen_questions: Set[str] = set()
        deduplicated = []
        
        for qa in qa_pairs:
            question = qa.get('question', '').strip().lower()
            
            # Skip if question is empty or already seen
            if not question or question in seen_questions:
                continue
            
            seen_questions.add(question)
            deduplicated.append(qa)
        
        removed_count = len(qa_pairs) - len(deduplicated)
        self.logger.info(f"Removed {removed_count} duplicate {language} Q&A pairs")
        
        return deduplicated
    
    def validate_qa_pair(self, qa: Dict) -> bool:
        """Validate that a Q&A pair has required fields and content."""
        required_fields = ['question', 'answer']
        
        for field in required_fields:
            if field not in qa or not qa[field] or not qa[field].strip():
                return False
        
        # Check minimum length
        if len(qa['question'].strip()) < 10 or len(qa['answer'].strip()) < 10:
            return False
        
        return True
    
    def filter_valid_qa_pairs(self, qa_pairs: List[Dict], language: str) -> List[Dict]:
        """Filter out invalid Q&A pairs."""
        self.logger.info(f"Filtering valid {language} Q&A pairs")
        
        valid_pairs = [qa for qa in qa_pairs if self.validate_qa_pair(qa)]
        
        invalid_count = len(qa_pairs) - len(valid_pairs)
        if invalid_count > 0:
            self.logger.warning(f"Filtered out {invalid_count} invalid {language} Q&A pairs")
        
        return valid_pairs
    
    def align_bilingual_qa(self, ar_qa_pairs: List[Dict], en_qa_pairs: List[Dict]) -> List[Dict]:
        """Align Arabic and English Q&A pairs by chunk ID."""
        self.logger.info("Aligning bilingual Q&A pairs")
        
        # Clean and deduplicate
        ar_qa_clean = self.filter_valid_qa_pairs(ar_qa_pairs, "arabic")
        en_qa_clean = self.filter_valid_qa_pairs(en_qa_pairs, "english")
        
        ar_qa_dedup = self.deduplicate_qa_pairs(ar_qa_clean, "arabic")
        en_qa_dedup = self.deduplicate_qa_pairs(en_qa_clean, "english")
        
        # Group by chunk ID
        ar_grouped = self.group_qa_by_chunk(ar_qa_dedup)
        en_grouped = self.group_qa_by_chunk(en_qa_dedup)
        
        # Align pairs
        aligned_pairs = []
        
        for chunk_id in sorted(set(ar_grouped.keys()) & set(en_grouped.keys())):
            ar_chunk_qa = ar_grouped[chunk_id]
            en_chunk_qa = en_grouped[chunk_id]
            
            # Pair Q&A from the same chunk
            min_pairs = min(len(ar_chunk_qa), len(en_chunk_qa))
            
            for i in range(min_pairs):
                aligned_pair = {
                    "question_ar": ar_chunk_qa[i]['question'],
                    "answer_ar": ar_chunk_qa[i]['answer'],
                    "question_en": en_chunk_qa[i]['question'],
                    "answer_en": en_chunk_qa[i]['answer'],
                    "chunk_id": chunk_id
                }
                aligned_pairs.append(aligned_pair)
        
        self.logger.info(f"Created {len(aligned_pairs)} aligned bilingual Q&A pairs")
        
        # Save intermediate result
        save_intermediate_file(aligned_pairs, "aligned_qa_pairs.json", "json")
        
        return aligned_pairs
    
    def final_quality_check(self, aligned_pairs: List[Dict]) -> List[Dict]:
        """Perform final quality check on aligned pairs."""
        self.logger.info("Performing final quality check")
        
        quality_pairs = []
        
        for pair in aligned_pairs:
            # Check all required fields exist and have content
            required_fields = ["question_ar", "answer_ar", "question_en", "answer_en"]
            
            if all(field in pair and pair[field] and pair[field].strip() for field in required_fields):
                # Check minimum lengths
                if (len(pair["question_ar"].strip()) >= 10 and 
                    len(pair["answer_ar"].strip()) >= 10 and
                    len(pair["question_en"].strip()) >= 10 and 
                    len(pair["answer_en"].strip()) >= 10):
                    
                    # Remove chunk_id from final output
                    final_pair = {
                        "question_ar": pair["question_ar"].strip(),
                        "answer_ar": pair["answer_ar"].strip(),
                        "question_en": pair["question_en"].strip(),
                        "answer_en": pair["answer_en"].strip()
                    }
                    quality_pairs.append(final_pair)
        
        removed_count = len(aligned_pairs) - len(quality_pairs)
        if removed_count > 0:
            self.logger.warning(f"Removed {removed_count} pairs during final quality check")
        
        self.logger.info(f"Final dataset contains {len(quality_pairs)} high-quality bilingual Q&A pairs")
        
        return quality_pairs
    
    def save_final_dataset(self, aligned_pairs: List[Dict], output_path: str) -> None:
        """Save the final bilingual Q&A dataset."""
        self.logger.info(f"Saving final dataset to {output_path}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for pair in aligned_pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + '\n')
            
            self.logger.info(f"Successfully saved {len(aligned_pairs)} Q&A pairs to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save final dataset: {e}")
            raise
    
    def merge_and_save(self, ar_qa_pairs: List[Dict], en_qa_pairs: List[Dict], output_path: str) -> None:
        """Complete merging pipeline."""
        self.logger.info("Starting data merging pipeline")
        
        # Align bilingual Q&A pairs
        aligned_pairs = self.align_bilingual_qa(ar_qa_pairs, en_qa_pairs)
        
        # Final quality check
        final_pairs = self.final_quality_check(aligned_pairs)
        
        # Save final dataset
        self.save_final_dataset(final_pairs, output_path)
        
        self.logger.info("Data merging pipeline completed")
