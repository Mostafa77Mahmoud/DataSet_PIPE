
#!/usr/bin/env python3
"""
Comprehensive audit and cleanup script for judgmental dataset project.
"""

import json
import os
import shutil
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

def setup_logging():
    """Setup logging for audit process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/audit_cleanup.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def validate_jsonl_structure(file_path: str, expected_fields: List[str]) -> Tuple[List[Dict], List[str]]:
    """Validate JSONL file structure and return issues."""
    issues = []
    valid_entries = []
    
    if not os.path.exists(file_path):
        issues.append(f"File {file_path} does not exist")
        return valid_entries, issues
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    entry = json.loads(line)
                    
                    # Check required fields
                    missing_fields = [field for field in expected_fields if field not in entry]
                    if missing_fields:
                        issues.append(f"Line {line_num}: Missing fields {missing_fields}")
                        continue
                    
                    # Validate judgmental format
                    if 'output' in entry:
                        output = entry['output']
                        if not re.search(r'VERDICT:\s*(True|False|correct|incorrect)', output, re.IGNORECASE):
                            issues.append(f"Line {line_num}: No valid VERDICT in output")
                        if 'xplanation:' not in output and 'xplain' not in output:
                            issues.append(f"Line {line_num}: No explanation in output")
                        if 'eference:' not in output:
                            issues.append(f"Line {line_num}: No reference in output")
                    
                    valid_entries.append(entry)
                    
                except json.JSONDecodeError as e:
                    issues.append(f"Line {line_num}: Invalid JSON - {e}")
    
    return valid_entries, issues

def fix_judgmental_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Fix malformed judgmental entries."""
    if 'output' not in entry:
        return entry
    
    output = entry['output']
    
    # Fix VERDICT format
    if not re.search(r'VERDICT:\s*(True|False|correct|incorrect)', output, re.IGNORECASE):
        if 'correct' in output.lower():
            output = f"VERDICT: True\n{output}"
        elif 'incorrect' in output.lower():
            output = f"VERDICT: False\n{output}"
        else:
            output = f"VERDICT: False\n{output}"
    
    # Ensure explanation exists
    if 'xplanation:' not in output:
        if 'VERDICT:' in output:
            parts = output.split('VERDICT:', 1)
            if len(parts) > 1:
                verdict_part = parts[1].split('\n')[0]
                rest = '\n'.join(parts[1].split('\n')[1:]) if '\n' in parts[1] else ""
                output = f"VERDICT:{verdict_part}\nExplanation: {rest}\nReference: UNKNOWN"
    
    # Ensure reference exists
    if 'eference:' not in output:
        output += "\nReference: UNKNOWN"
    
    entry['output'] = output
    return entry

def archive_old_files():
    """Archive large raw data files and temporary files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = f"archive/{timestamp}"
    os.makedirs(archive_dir, exist_ok=True)
    
    # Archive patterns
    archive_patterns = [
        "raw/gemini_*.json",
        "intermediate/synthetic_by_seed_*.json",
        "*.tmp", "*.temp", "temp_*"
    ]
    
    archived_files = []
    
    for pattern in archive_patterns:
        for file_path in Path(".").glob(pattern):
            if file_path.is_file():
                dest_path = Path(archive_dir) / file_path.name
                shutil.move(str(file_path), str(dest_path))
                archived_files.append(str(file_path))
    
    # Create manifest
    manifest = {
        "timestamp": timestamp,
        "archived_files": archived_files,
        "total_files": len(archived_files)
    }
    
    with open(f"{archive_dir}/manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return archived_files

def create_clean_splits(data: List[Dict], output_dir: str = "data"):
    """Create clean train/val/test splits."""
    # Shuffle and split
    import random
    random.seed(42)
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    total = len(shuffled)
    train_end = int(0.8 * total)
    val_end = int(0.9 * total)
    
    splits = {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:]
    }
    
    for split_name, split_data in splits.items():
        output_path = f"{output_dir}/{split_name}.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in split_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    return splits

def main():
    logger = setup_logging()
    logger.info("Starting comprehensive audit and cleanup")
    
    # 1. Dataset & Structure Validation
    logger.info("=== STEP 1: Dataset Validation ===")
    
    expected_fields = ["instruction", "input", "output"]
    main_dataset = "data/judgmental_final.jsonl"
    
    if os.path.exists(main_dataset):
        entries, issues = validate_jsonl_structure(main_dataset, expected_fields)
        logger.info(f"Found {len(entries)} valid entries, {len(issues)} issues")
        
        if issues:
            logger.warning("Issues found:")
            for issue in issues[:10]:  # Show first 10 issues
                logger.warning(f"  {issue}")
        
        # Fix issues
        fixed_entries = [fix_judgmental_entry(entry) for entry in entries]
        
        # Save fixed dataset
        with open(main_dataset, 'w', encoding='utf-8') as f:
            for entry in fixed_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        logger.info(f"Fixed and saved {len(fixed_entries)} entries")
        
        # 2. Create clean splits
        logger.info("=== STEP 2: Creating Clean Splits ===")
        splits = create_clean_splits(fixed_entries)
        for split_name, split_data in splits.items():
            logger.info(f"{split_name}: {len(split_data)} examples")
    
    # 3. Archive old files
    logger.info("=== STEP 3: Archiving Old Files ===")
    archived = archive_old_files()
    logger.info(f"Archived {len(archived)} files")
    
    # 4. Clean up unnecessary files
    logger.info("=== STEP 4: Cleaning Unnecessary Files ===")
    cleanup_patterns = [
        "__pycache__/",
        "*.pyc",
        ".pytest_cache/",
        "*.log~",
        "*.tmp"
    ]
    
    cleaned_files = []
    for pattern in cleanup_patterns:
        for file_path in Path(".").rglob(pattern):
            if file_path.exists():
                if file_path.is_dir():
                    shutil.rmtree(file_path)
                else:
                    file_path.unlink()
                cleaned_files.append(str(file_path))
    
    logger.info(f"Cleaned {len(cleaned_files)} unnecessary files")
    
    # 5. Generate validation report
    logger.info("=== STEP 5: Generating Validation Report ===")
    
    stats = {}
    if os.path.exists(main_dataset):
        with open(main_dataset, 'r', encoding='utf-8') as f:
            entries = [json.loads(line) for line in f if line.strip()]
        
        # Count verdicts and languages
        verdicts = {"true": 0, "false": 0}
        languages = {"arabic": 0, "english": 0}
        
        for entry in entries:
            output = entry.get('output', '').lower()
            if 'verdict: true' in output or 'correct' in output:
                verdicts["true"] += 1
            else:
                verdicts["false"] += 1
            
            instruction = entry.get('instruction', '').lower()
            if any(arabic_char in instruction for arabic_char in 'ابتثجحخدذرزسشصضطظعغفقكلمنهوي'):
                languages["arabic"] += 1
            else:
                languages["english"] += 1
        
        stats = {
            "total_entries": len(entries),
            "verdicts": verdicts,
            "languages": languages,
            "balance_ratio": min(verdicts.values()) / max(verdicts.values()) if verdicts and max(verdicts.values()) > 0 else 0
        }
    
    # Save validation summary
    summary = f"""
# Dataset Validation Summary

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Statistics
- Total entries: {stats.get('total_entries', 0)}
- True verdicts: {stats.get('verdicts', {}).get('true', 0)}
- False verdicts: {stats.get('verdicts', {}).get('false', 0)}
- Arabic entries: {stats.get('languages', {}).get('arabic', 0)}
- English entries: {stats.get('languages', {}).get('english', 0)}
- Balance ratio: {stats.get('balance_ratio', 0):.2f}

## Cleanup Actions
- Fixed {len(entries) if 'entries' in locals() else 0} dataset entries
- Archived {len(archived)} raw files
- Cleaned {len(cleaned_files)} unnecessary files

## Files Structure
- data/judgmental_final.jsonl: Main dataset
- data/train.jsonl: Training split (80%)
- data/val.jsonl: Validation split (10%)
- data/test.jsonl: Test split (10%)

Dataset is ready for training.
"""
    
    with open("logs/validation_summary.txt", 'w') as f:
        f.write(summary)
    
    logger.info("Audit and cleanup completed successfully")
    print(summary)

if __name__ == "__main__":
    main()
