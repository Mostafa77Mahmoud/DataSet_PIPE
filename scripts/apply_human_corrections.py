
#!/usr/bin/env python3
"""
Apply human corrections to judgmental dataset
"""

import json
import csv
import argparse

def load_jsonl(file_path):
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    """Save data to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_corrections(csv_file):
    """Load human corrections from CSV."""
    corrections = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['reviewer_verdict']:  # Only process if reviewer provided verdict
                corrections[row['seed_id']] = {
                    'verdict': row['reviewer_verdict'],
                    'notes': row['reviewer_notes']
                }
    return corrections

def main():
    parser = argparse.ArgumentParser(description="Apply human corrections")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--review", required=True, help="Human review CSV file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    args = parser.parse_args()
    
    # Load data and corrections
    data = load_jsonl(args.input)
    corrections = load_corrections(args.review)
    
    print(f"Loaded {len(data)} examples and {len(corrections)} corrections")
    
    # Apply corrections
    corrected_count = 0
    for item in data:
        seed_id = item.get('meta', {}).get('seed_id', '')
        if seed_id in corrections:
            correction = corrections[seed_id]
            
            # Update verdict in output
            new_verdict = correction['verdict']
            output = item['output']
            if 'VERDICT:' in output:
                old_verdict = "True" if "VERDICT: True" in output else "False"
                if old_verdict != new_verdict:
                    item['output'] = output.replace(f"VERDICT: {old_verdict}", f"VERDICT: {new_verdict}")
                    item['meta']['human_verified'] = True
                    item['meta']['human_correction'] = {
                        'original_verdict': old_verdict,
                        'corrected_verdict': new_verdict,
                        'notes': correction['notes']
                    }
                    corrected_count += 1
    
    # Save corrected dataset
    save_jsonl(data, args.output)
    print(f"Applied {corrected_count} corrections and saved to {args.output}")

if __name__ == "__main__":
    main()
