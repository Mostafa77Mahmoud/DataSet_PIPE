#!/usr/bin/env python3
"""
Create human review sample from judgmental dataset.
"""

import json
import os
import random
import csv
from datetime import datetime
from typing import List, Dict, Any
import logging
import subprocess # Imported subprocess for get_git_commit_hash

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_human_review_sample(
    dataset_path: str,
    sample_size: int = None # Changed default to None for dynamic calculation
) -> None:
    """Create a human review sample from the dataset."""

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset {dataset_path} not found")
        print(f"Error: Dataset {dataset_path} not found")
        return

    # Load dataset
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            entries = [json.loads(line) for line in f if line.strip()]
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {dataset_path}: {e}")
        print(f"Error: Could not parse dataset file {dataset_path}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {dataset_path}: {e}")
        print(f"An unexpected error occurred while loading the dataset.")
        return


    logger.info(f"Loaded {len(entries)} entries from {dataset_path}")

    # Safety check for empty dataset
    total = len(entries)
    if total == 0:
        logger.warning("No entries found in dataset. Skipping human review sample creation.")
        print("⚠️ No entries found - cannot create human review sample")
        with open("logs/validation_summary.txt", "a") as f:
            f.write(f"\n⚠️ Human review sample creation skipped: dataset empty ({dataset_path})\n")
        return

    # Dynamic sample size calculation
    if sample_size is None:
        sample_size = min(max(int(total * 0.05), 1), min(200, total)) # Default to 5%, capped at 200

    # Handle small datasets
    if total < 20:
        logger.info(f"Small dataset ({total} entries) - using entire dataset for review")
        sample_entries = entries
        sample_size = total
        print(f"⚠️ Small dataset ({total} entries), using entire dataset for review")
    else:
        # Create random sample
        random.seed(42) # Seed for reproducibility
        sample_entries = random.sample(entries, sample_size)

    logger.info(f"Creating sample of {sample_size} entries")

    # Create review directory
    output_dir = "review" # Defined output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save sample as JSONL
    sample_jsonl_path = f"{output_dir}/sample_for_human_review.jsonl"
    try:
        with open(sample_jsonl_path, 'w', encoding='utf-8') as f:
            for i, entry in enumerate(sample_entries):
                # Add sample ID for tracking
                entry_with_id = {**entry, "sample_id": i}
                f.write(json.dumps(entry_with_id, ensure_ascii=False) + '\n')
        logger.info(f"Sample saved to {sample_jsonl_path}")
    except IOError as e:
        logger.error(f"Error writing sample JSONL to {sample_jsonl_path}: {e}")
        print(f"Error: Could not save the review sample.")
        return

    # Create CSV template for human review
    csv_template_path = f"{output_dir}/human_review_template.csv"

    try:
        with open(csv_template_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'sample_id',
                'claim_preview',
                'model_verdict',
                'reviewer_verdict',
                'reviewer_explanation',
                'quality_score',
                'notes'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i, entry in enumerate(sample_entries):
                # Extract claim preview (first 100 chars)
                instruction = entry.get('instruction', '')
                claim_start = -1
                if 'Claim:' in instruction:
                    claim_start = instruction.find('Claim:')
                elif 'الادعاء:' in instruction:
                    claim_start = instruction.find('الادعاء:')

                if claim_start != -1:
                    claim_preview = instruction[claim_start:claim_start+100] + "..."
                else:
                    claim_preview = instruction[:100] + "..."

                # Extract model verdict
                output = entry.get('output', '')
                if 'VERDICT: True' in output:
                    model_verdict = 'True'
                elif 'VERDICT: False' in output:
                    model_verdict = 'False'
                else:
                    model_verdict = 'Unknown'

                writer.writerow({
                    'sample_id': i,
                    'claim_preview': claim_preview.replace('\n', ' '),
                    'model_verdict': model_verdict,
                    'reviewer_verdict': '',  # To be filled by human reviewer
                    'reviewer_explanation': '',  # To be filled by human reviewer
                    'quality_score': '',  # 1-5 scale, to be filled by human reviewer
                    'notes': ''  # Additional notes
                })
        logger.info(f"CSV template saved to {csv_template_path}")
    except IOError as e:
        logger.error(f"Error writing CSV template to {csv_template_path}: {e}")
        print(f"Error: Could not save the CSV template.")
        return

    # Create snapshot info
    snapshot_info = {
        "dataset_path": dataset_path,
        "sample_size": sample_size,
        "sample_percentage": 0.05 if sample_size == int(len(entries) * 0.05) else (sample_size/total if total > 0 else 0), # Calculate percentage dynamically
        "total_entries": len(entries),
        "created_at": datetime.now().isoformat(),
        "git_commit": get_git_commit_hash(),
        "instructions": {
            "csv_fields": {
                "sample_id": "Unique identifier for tracking",
                "claim_preview": "Preview of the claim being verified",
                "model_verdict": "Model's verdict (True/False)",
                "reviewer_verdict": "Human reviewer's verdict (True/False/Uncertain)",
                "reviewer_explanation": "Explanation for any disagreement",
                "quality_score": "Overall quality score (1-5, where 5 is excellent)",
                "notes": "Additional notes or observations"
            },
            "review_guidelines": [
                "Check if the model's verdict matches the context and claim",
                "Verify that references are accurate (not fabricated)",
                "Assess explanation quality and clarity",
                "Flag any potential issues or improvements"
            ]
        }
    }

    snapshot_path = f"{output_dir}/snapshot_info.json"
    try:
        with open(snapshot_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot_info, f, ensure_ascii=False, indent=2)
        logger.info(f"Snapshot info saved to {snapshot_path}")
    except IOError as e:
        logger.error(f"Error writing snapshot info to {snapshot_path}: {e}")
        print(f"Error: Could not save snapshot information.")
        return

    print(f"\n🎉 Human review sample created successfully!")
    print(f"Review the CSV template at: {csv_template_path}")

def get_git_commit_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            cwd='.',
            check=True
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except FileNotFoundError:
        logger.warning("Git command not found. Cannot retrieve commit hash.")
        return "unknown"
    except subprocess.CalledProcessError:
        logger.warning("Failed to execute git command. Cannot retrieve commit hash.")
        return "unknown"
    except Exception as e:
        logger.error(f"An unexpected error occurred while getting git commit hash: {e}")
        return "unknown"

def main():
    """Main function."""

    # Find the main dataset
    dataset_paths = [
        "data/judgmental_final.jsonl",
        "data/synthetic_judgmental.jsonl",
        "data/preview_first_200.jsonl"
    ]

    dataset_path = None
    for path in dataset_paths:
        if os.path.exists(path):
            dataset_path = path
            break

    if not dataset_path:
        logger.error("No judgmental dataset found in expected locations.")
        print("Error: No judgmental dataset found")
        return

    create_human_review_sample(dataset_path)

if __name__ == "__main__":
    main()