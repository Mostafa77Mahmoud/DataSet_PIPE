
#!/usr/bin/env python3
"""
Archive old JSON/JSONL files with safety checks.
"""

import os
import json
import shutil
import glob
from datetime import datetime, timedelta
from pathlib import Path
import logging

def setup_logging():
    """Setup logging."""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def get_essential_files() -> set:
    """Get list of essential files to keep."""
    return {
        "data/judgmental_final.jsonl",
        "data/train.jsonl",
        "data/val.jsonl", 
        "data/test.jsonl",
        "output/judgmental_alpaca.jsonl",
        "data/metadata.json",
        "intermediate/aaofi_cleaned.txt",
        "intermediate/english_cleaned.txt",
        "intermediate/arabic_cleaned.txt"
    }

def is_file_referenced(file_path: str, metadata_file: str, validation_file: str) -> bool:
    """Check if file is referenced in metadata or validation files."""
    file_name = os.path.basename(file_path)
    
    # Check metadata.json
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                if file_name in str(metadata):
                    return True
        except:
            pass
    
    # Check validation files
    for val_file in glob.glob("logs/validation_report*.json"):
        try:
            with open(val_file, 'r') as f:
                content = f.read()
                if file_name in content:
                    return True
        except:
            pass
    
    return False

def archive_old_files(permanent_delete_days: int = 30):
    """Archive old JSON/JSONL files."""
    logger = setup_logging()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = f"archive/{timestamp}"
    
    os.makedirs(archive_dir, exist_ok=True)
    
    essential_files = get_essential_files()
    manifest = {
        "timestamp": timestamp,
        "archived_files": [],
        "kept_files": [],
        "deleted_files": []
    }
    
    # Find all JSON/JSONL files
    patterns = ["**/*.json", "**/*.jsonl"]
    candidates = []
    
    for pattern in patterns:
        candidates.extend(Path(".").glob(pattern))
    
    logger.info(f"Found {len(candidates)} JSON/JSONL files to evaluate")
    
    for file_path in candidates:
        rel_path = str(file_path)
        
        # Skip essential files
        if rel_path in essential_files:
            manifest["kept_files"].append({
                "path": rel_path,
                "reason": "essential"
            })
            continue
        
        # Skip if already in archive
        if rel_path.startswith("archive/"):
            continue
        
        # Check if referenced
        if is_file_referenced(rel_path, "data/metadata.json", "logs/validation_report.json"):
            manifest["kept_files"].append({
                "path": rel_path,
                "reason": "referenced"
            })
            continue
        
        # Archive the file
        archive_path = os.path.join(archive_dir, rel_path)
        os.makedirs(os.path.dirname(archive_path), exist_ok=True)
        
        # Get file info before moving
        file_stat = file_path.stat()
        file_info = {
            "original_path": rel_path,
            "archived_path": archive_path,
            "size": file_stat.st_size,
            "mtime": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            "archived_at": datetime.now().isoformat()
        }
        
        # Move file to archive
        shutil.move(str(file_path), archive_path)
        manifest["archived_files"].append(file_info)
        logger.info(f"Archived: {rel_path}")
    
    # Handle permanent deletion of old archived files
    cutoff_date = datetime.now() - timedelta(days=permanent_delete_days)
    
    for archive_subdir in glob.glob("archive/*/"):
        try:
            dir_timestamp = os.path.basename(archive_subdir.rstrip('/'))
            dir_date = datetime.strptime(dir_timestamp, "%Y%m%d_%H%M%S")
            
            if dir_date < cutoff_date:
                # Check if any files in this archive are still referenced
                manifest_file = os.path.join(archive_subdir, "manifest.json")
                if os.path.exists(manifest_file):
                    with open(manifest_file, 'r') as f:
                        old_manifest = json.load(f)
                    
                    # Only delete if no files are marked as referenced
                    can_delete = True
                    for archived_file in old_manifest.get("archived_files", []):
                        if is_file_referenced(archived_file["original_path"], "data/metadata.json", "logs/validation_report.json"):
                            can_delete = False
                            break
                    
                    if can_delete:
                        shutil.rmtree(archive_subdir)
                        manifest["deleted_files"].append({
                            "archive_path": archive_subdir,
                            "deleted_at": datetime.now().isoformat(),
                            "reason": f"older than {permanent_delete_days} days"
                        })
                        logger.info(f"Permanently deleted old archive: {archive_subdir}")
        except:
            continue
    
    # Save manifest
    manifest_path = f"archive/manifest_{timestamp}.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Archival complete. Manifest saved to {manifest_path}")
    logger.info(f"Archived: {len(manifest['archived_files'])} files")
    logger.info(f"Kept: {len(manifest['kept_files'])} files")
    logger.info(f"Deleted: {len(manifest['deleted_files'])} archives")
    
    return manifest_path

if __name__ == "__main__":
    archive_old_files()
