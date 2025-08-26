
#!/usr/bin/env python3
"""
Archive unnecessary files safely
"""

import os
import json
import shutil
from datetime import datetime
import glob

def create_archive_dir():
    """Create timestamped archive directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = f"archive/{timestamp}"
    os.makedirs(archive_dir, exist_ok=True)
    return archive_dir, timestamp

def is_file_referenced(file_path, metadata_file, validation_file):
    """Check if file is referenced in metadata or validation reports."""
    try:
        # Check metadata
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                if file_path in str(metadata):
                    return True
        
        # Check validation report
        if os.path.exists(validation_file):
            with open(validation_file, 'r') as f:
                validation = json.load(f)
                if file_path in str(validation):
                    return True
        
        return False
    except:
        return True  # Safe default - keep file if can't determine

def main():
    archive_dir, timestamp = create_archive_dir()
    manifest = {
        "timestamp": timestamp,
        "archived_files": [],
        "deleted_files": []
    }
    
    print(f"Creating archive directory: {archive_dir}")
    
    # Files to potentially archive
    candidates = []
    
    # Old raw files (keep recent ones)
    raw_files = glob.glob("raw/*.json")
    if len(raw_files) > 100:  # If too many, archive oldest
        raw_files.sort(key=os.path.getmtime)
        candidates.extend(raw_files[:-50])  # Keep latest 50
    
    # Temporary files
    candidates.extend(glob.glob("intermediate/*_tmp*"))
    candidates.extend(glob.glob("**/*.tmp", recursive=True))
    
    # Archive candidates
    metadata_file = "data/metadata.json"
    validation_file = "logs/validation_report.json"
    
    for file_path in candidates:
        if os.path.exists(file_path):
            if not is_file_referenced(file_path, metadata_file, validation_file):
                # Archive the file
                rel_path = os.path.relpath(file_path)
                archive_path = os.path.join(archive_dir, rel_path)
                os.makedirs(os.path.dirname(archive_path), exist_ok=True)
                shutil.move(file_path, archive_path)
                manifest["archived_files"].append({
                    "original": rel_path,
                    "archived": archive_path
                })
                print(f"Archived: {rel_path}")
    
    # Delete safe files
    safe_delete_patterns = [
        "**/*.pyc",
        "**/__pycache__",
        "**/.pytest_cache"
    ]
    
    for pattern in safe_delete_patterns:
        for file_path in glob.glob(pattern, recursive=True):
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
            manifest["deleted_files"].append(file_path)
            print(f"Deleted: {file_path}")
    
    # Save manifest
    manifest_file = f"archive/manifest_{timestamp}.json"
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Archive manifest saved to: {manifest_file}")
    print(f"Archived {len(manifest['archived_files'])} files")
    print(f"Deleted {len(manifest['deleted_files'])} files")

if __name__ == "__main__":
    main()
