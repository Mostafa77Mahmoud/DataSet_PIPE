
#!/usr/bin/env python3
"""
Convert judgmental dataset to Alpaca format JSONL file with validation.
"""

import json
import os
import re
from typing import List, Dict, Any

def validate_and_fix_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and fix a judgmental entry for Alpaca format."""
    
    # Extract components
    claim = entry.get('claim', '')
    context = entry.get('context', '')
    explanation = entry.get('explanation', '')
    entry_type = entry.get('type', '')
    reference = entry.get('reference', 'UNKNOWN')
    language = entry.get('language', 'english')
    
    # Create instruction
    if language == 'arabic':
        instruction = f"بناءً على السياق التالي، حدد ما إذا كان هذا الادعاء صحيحًا أم خاطئًا وأوضح السبب:\n\nالسياق: {context}\n\nالادعاء: {claim}"
    else:
        instruction = f"Based on the following context, determine if this claim is correct or incorrect and explain why:\n\nContext: {context}\n\nClaim: {claim}"
    
    # Determine verdict
    verdict = "True" if entry_type == 'correct' else "False"
    
    # Create properly formatted output
    output = f"VERDICT: {verdict}\nExplanation: {explanation}\nReference: {reference}"
    
    return {
        "instruction": instruction,
        "input": "",
        "output": output
    }

def convert_judgmental_to_alpaca(judgmental_file: str, output_file: str) -> None:
    """Convert judgmental dataset to Alpaca format with validation."""
    
    alpaca_data = []
    
    print(f"Loading judgmental data from: {judgmental_file}")
    
    if not os.path.exists(judgmental_file):
        print(f"Error: File {judgmental_file} does not exist")
        return
    
    with open(judgmental_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data = json.loads(line)
                    
                    # Check if already in Alpaca format
                    if all(field in data for field in ['instruction', 'input', 'output']):
                        # Validate existing Alpaca entry
                        output = data['output']
                        if not re.search(r'VERDICT:\s*(True|False)', output):
                            print(f"Warning: Line {line_num}: Missing proper VERDICT format")
                            continue
                        alpaca_data.append(data)
                    else:
                        # Convert from raw judgmental format
                        alpaca_entry = validate_and_fix_entry(data)
                        alpaca_data.append(alpaca_entry)
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON at line {line_num}: {e}")
                    continue
    
    print(f"Converted {len(alpaca_data)} judgmental entries to Alpaca format")
    
    # Save Alpaca format
    os.makedirs("output", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in alpaca_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Saved Alpaca format file to: {output_file}")
    
    # Display statistics
    file_size = os.path.getsize(output_file) / 1024  # KB
    print(f"File size: {file_size:.1f} KB")
    
    # Validate final output
    verdicts = {"true": 0, "false": 0}
    languages = {"arabic": 0, "english": 0}
    
    for entry in alpaca_data:
        output = entry['output'].lower()
        if 'verdict: true' in output:
            verdicts["true"] += 1
        else:
            verdicts["false"] += 1
        
        instruction = entry['instruction']
        if any(arabic_char in instruction for arabic_char in 'ابتثجحخدذرزسشصضطظعغفقكلمنهوي'):
            languages["arabic"] += 1
        else:
            languages["english"] += 1
    
    print(f"Validation: {verdicts['true']} True, {verdicts['false']} False")
    print(f"Languages: {languages['arabic']} Arabic, {languages['english']} English")

def main():
    """Main function to convert judgmental data."""
    
    # Check if judgmental data exists
    judgmental_files = [
        "data/judgmental_final.jsonl",
        "data/synthetic_judgmental.jsonl", 
        "data/preview_first_200.jsonl"
    ]
    
    input_file = None
    for file_path in judgmental_files:
        if os.path.exists(file_path):
            input_file = file_path
            break
    
    if not input_file:
        print("Error: No judgmental dataset found. Please run generate_synthetic.py first.")
        return
    
    output_file = "output/judgmental_alpaca.jsonl"
    
    convert_judgmental_to_alpaca(input_file, output_file)
    
    print("\n🎉 Judgmental to Alpaca conversion completed!")

if __name__ == "__main__":
    main()
