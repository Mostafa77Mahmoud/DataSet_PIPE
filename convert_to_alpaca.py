#!/usr/bin/env python3
"""
Convert intermediate Q&A JSON files to Alpaca format JSONL files.

Alpaca format structure:
{
    "instruction": "question here",
    "input": "",
    "output": "answer here"
}
"""

import json
import os
from typing import List, Dict, Any

def convert_qa_pairs_to_alpaca(qa_pairs: List[Dict], language: str) -> List[Dict]:
    """Convert Q&A pairs to Alpaca format."""
    alpaca_data = []
    
    for qa in qa_pairs:
        alpaca_entry = {
            "instruction": qa.get("question", ""),
            "input": "",
            "output": qa.get("answer", "")
        }
        alpaca_data.append(alpaca_entry)
    
    return alpaca_data

def convert_aligned_pairs_to_alpaca(aligned_pairs: List[Dict]) -> List[Dict]:
    """Convert aligned bilingual pairs to separate Alpaca entries."""
    alpaca_data = []
    
    for pair in aligned_pairs:
        # Arabic Q&A
        if "question_ar" in pair and "answer_ar" in pair:
            arabic_entry = {
                "instruction": pair["question_ar"],
                "input": "",
                "output": pair["answer_ar"]
            }
            alpaca_data.append(arabic_entry)
        
        # English Q&A
        if "question_en" in pair and "answer_en" in pair:
            english_entry = {
                "instruction": pair["question_en"],
                "input": "",
                "output": pair["answer_en"]
            }
            alpaca_data.append(english_entry)
    
    return alpaca_data

def save_alpaca_jsonl(data: List[Dict], filename: str) -> None:
    """Save data in JSONL format."""
    os.makedirs("output", exist_ok=True)
    filepath = os.path.join("output", filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✓ Saved {len(data)} entries to {filepath}")

def main():
    """Convert all intermediate files to Alpaca format."""
    print("Converting intermediate files to Alpaca format JSONL...")
    
    # Convert Arabic Q&A pairs
    print("\n1. Converting Arabic Q&A pairs...")
    try:
        with open("intermediate/arabic_qa_pairs.json", 'r', encoding='utf-8') as f:
            arabic_qa = json.load(f)
        
        arabic_alpaca = convert_qa_pairs_to_alpaca(arabic_qa, "arabic")
        save_alpaca_jsonl(arabic_alpaca, "arabic_qa_alpaca.jsonl")
    except FileNotFoundError:
        print("❌ Arabic Q&A pairs file not found")
    except Exception as e:
        print(f"❌ Error converting Arabic Q&A: {e}")
    
    # Convert English Q&A pairs
    print("\n2. Converting English Q&A pairs...")
    try:
        with open("intermediate/english_qa_pairs.json", 'r', encoding='utf-8') as f:
            english_qa = json.load(f)
        
        english_alpaca = convert_qa_pairs_to_alpaca(english_qa, "english")
        save_alpaca_jsonl(english_alpaca, "english_qa_alpaca.jsonl")
    except FileNotFoundError:
        print("❌ English Q&A pairs file not found")
    except Exception as e:
        print(f"❌ Error converting English Q&A: {e}")
    
    # Convert aligned bilingual pairs - PRIORITY CONVERSION
    print("\n3. Converting aligned bilingual pairs (MAIN DATASET)...")
    try:
        with open("intermediate/aligned_qa_pairs.json", 'r', encoding='utf-8') as f:
            aligned_qa = json.load(f)
        
        print(f"✓ Loaded {len(aligned_qa)} aligned Q&A pairs")
        
        # Convert to separate Arabic and English entries
        aligned_alpaca = convert_aligned_pairs_to_alpaca(aligned_qa)
        save_alpaca_jsonl(aligned_alpaca, "bilingual_qa_alpaca.jsonl")
        
        # Also create separate Arabic and English files from aligned data
        arabic_only = []
        english_only = []
        
        for pair in aligned_qa:
            if "question_ar" in pair and "answer_ar" in pair:
                arabic_only.append({
                    "instruction": pair["question_ar"],
                    "input": "",
                    "output": pair["answer_ar"]
                })
            
            if "question_en" in pair and "answer_en" in pair:
                english_only.append({
                    "instruction": pair["question_en"],
                    "input": "",
                    "output": pair["answer_en"]
                })
        
        save_alpaca_jsonl(arabic_only, "arabic_aligned_alpaca.jsonl")
        save_alpaca_jsonl(english_only, "english_aligned_alpaca.jsonl")
        
    except FileNotFoundError:
        print("❌ Aligned Q&A pairs file not found")
    except Exception as e:
        print(f"❌ Error converting aligned Q&A: {e}")
    
    print("\n🎉 Conversion completed! All files saved in output/ directory.")
    
    # Display file statistics
    print("\n📊 File Statistics:")
    output_files = [
        ("arabic_qa_alpaca.jsonl", "Arabic Q&A (Alpaca format)"),
        ("english_qa_alpaca.jsonl", "English Q&A (Alpaca format)"),
        ("bilingual_qa_alpaca.jsonl", "Bilingual Combined (Alpaca format)"),
        ("arabic_aligned_alpaca.jsonl", "Arabic from Aligned Pairs"),
        ("english_aligned_alpaca.jsonl", "English from Aligned Pairs")
    ]
    
    for filename, description in output_files:
        filepath = os.path.join("output", filename)
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / 1024  # KB
            with open(filepath, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            print(f"  • {description}: {line_count} entries, {file_size:.1f} KB")

if __name__ == "__main__":
    main()