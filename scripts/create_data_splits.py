
#!/usr/bin/env python3
"""
Create balanced train/val/test splits
"""

import json
import random
from collections import Counter

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

def create_balanced_splits(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Create balanced splits preserving class distribution."""
    # Separate by type and language
    groups = {}
    for item in data:
        meta = item.get('meta', {})
        key = (meta.get('type', 'unknown'), meta.get('language', 'unknown'))
        if key not in groups:
            groups[key] = []
        groups[key].append(item)
    
    train_data, val_data, test_data = [], [], []
    
    # Split each group proportionally
    for key, items in groups.items():
        random.shuffle(items)
        n = len(items)
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_data.extend(items[:train_end])
        val_data.extend(items[train_end:val_end])
        test_data.extend(items[val_end:])
    
    # Shuffle final splits
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data

def main():
    # Load dataset
    data = load_jsonl('data/judgmental_final.jsonl')
    print(f"Loaded {len(data)} examples")
    
    # Check distribution
    types = [item.get('meta', {}).get('type', 'unknown') for item in data]
    languages = [item.get('meta', {}).get('language', 'unknown') for item in data]
    
    print("Type distribution:", Counter(types))
    print("Language distribution:", Counter(languages))
    
    # Create splits
    random.seed(42)
    train_data, val_data, test_data = create_balanced_splits(data)
    
    # Save splits
    save_jsonl(train_data, 'data/train.jsonl')
    save_jsonl(val_data, 'data/val.jsonl')
    save_jsonl(test_data, 'data/test.jsonl')
    
    print(f"Created splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Verify balance
    for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        types = [item.get('meta', {}).get('type', 'unknown') for item in split_data]
        languages = [item.get('meta', {}).get('language', 'unknown') for item in split_data]
        print(f"{split_name} - Types: {Counter(types)}, Languages: {Counter(languages)}")

if __name__ == "__main__":
    main()
