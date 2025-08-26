
#!/usr/bin/env python3
"""
Convert aligned bilingual Q&A pairs into judgmental seed format.
"""

import json
import os
import sys
import argparse
from typing import List, Dict, Any
import hashlib

def load_aligned_qa(file_path: str) -> List[Dict]:
    """Load aligned Q&A pairs from JSON or JSONL file."""
    data = []
    
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    return data

def create_judgmental_seed(aligned_pairs: List[Dict]) -> List[Dict]:
    """Convert aligned pairs to judgmental seed format."""
    seed_data = []
    
    for i, pair in enumerate(aligned_pairs):
        # Arabic seed
        if "question_ar" in pair and "answer_ar" in pair:
            ar_claim = f"السؤال: {pair['question_ar']} الجواب: {pair['answer_ar']}"
            ar_seed = {
                "seed_id": f"ar_{i:04d}",
                "language": "ar",
                "claim": ar_claim,
                "context": pair.get("answer_ar", "")[:500],  # Use answer as context
                "meta": {
                    "chunk_id": pair.get("chunk_id", i),
                    "source": "aligned",
                    "aligned_id": i,
                    "original_question": pair["question_ar"],
                    "original_answer": pair["answer_ar"]
                }
            }
            seed_data.append(ar_seed)
        
        # English seed
        if "question_en" in pair and "answer_en" in pair:
            en_claim = f"Question: {pair['question_en']} Answer: {pair['answer_en']}"
            en_seed = {
                "seed_id": f"en_{i:04d}",
                "language": "en", 
                "claim": en_claim,
                "context": pair.get("answer_en", "")[:500],  # Use answer as context
                "meta": {
                    "chunk_id": pair.get("chunk_id", i),
                    "source": "aligned",
                    "aligned_id": i,
                    "original_question": pair["question_en"],
                    "original_answer": pair["answer_en"]
                }
            }
            seed_data.append(en_seed)
    
    return seed_data

def save_seed_file(seed_data: List[Dict], output_path: str) -> None:
    """Save seed data to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for seed in seed_data:
            f.write(json.dumps(seed, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Convert aligned Q&A to judgmental seed format")
    parser.add_argument("--input", default="intermediate/aligned_qa_pairs.json", help="Input aligned Q&A file")
    parser.add_argument("--output", default="data/judgmental_seed.jsonl", help="Output seed file")
    
    args = parser.parse_args()
    
    print(f"Loading aligned Q&A pairs from: {args.input}")
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        sys.exit(1)
    
    # Load aligned pairs
    aligned_pairs = load_aligned_qa(args.input)
    print(f"Loaded {len(aligned_pairs)} aligned Q&A pairs")
    
    # Convert to seed format
    seed_data = create_judgmental_seed(aligned_pairs)
    print(f"Created {len(seed_data)} seed entries")
    
    # Save seed file
    save_seed_file(seed_data, args.output)
    print(f"Saved seed file to: {args.output}")
    
    # Print statistics
    ar_count = sum(1 for s in seed_data if s["language"] == "ar")
    en_count = sum(1 for s in seed_data if s["language"] == "en")
    print(f"Language distribution: Arabic={ar_count}, English={en_count}")

if __name__ == "__main__":
    main()
