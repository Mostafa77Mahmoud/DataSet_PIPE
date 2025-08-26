#!/usr/bin/env python3
"""
Master script to run full audit, cleanup, and validation.
"""

import os
import subprocess
import sys
from datetime import datetime

def run_command(cmd, description):
    """Run a command and log the result."""
    print(f"\n{'='*50}")
    print(f"RUNNING: {description}")
    print(f"{'='*50}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"❌ FAILED: {description}")
            if result.stderr:
                print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ ERROR: {description} - {e}")
        return False

def main():
    """Run full audit and cleanup."""

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_file = f"logs/audit_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    print("🔍 STARTING COMPREHENSIVE AUDIT AND CLEANUP")
    print(f"Timestamp: {timestamp}")
    print(f"Log file: {log_file}")

    # Ensure required directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("review", exist_ok=True)
    os.makedirs("ingest", exist_ok=True)
    os.makedirs("synthesis", exist_ok=True)

    # Step 0: Check if dataset needs generation
    current_size = 0
    if os.path.exists("data/judgmental_final.jsonl"):
        try:
            with open("data/judgmental_final.jsonl", 'r', encoding='utf-8') as f:
                current_size = sum(1 for line in f if line.strip())
        except Exception as e:
            print(f"Warning: Could not read dataset file: {e}")
            current_size = 0

    print(f"Current dataset size: {current_size} entries")

    # Try to auto-generate if dataset is too small (only if dependencies available)
    if current_size < 1000:
        print(f"Dataset too small ({current_size} < 1000), attempting to generate synthetic data...")
        # Check if google.generativeai is available
        import subprocess
        try:
            result = subprocess.run([
                "python", "-c", "import google.generativeai as genai; import os; print('OK' if os.getenv('GEMINI_API_KEY') else 'NO_KEY')"
            ], capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip() == 'OK':
                success_gen = run_command(
                    f"python synthesis/generate_synthetic.py --target-size 2000 --preview --concurrency 2",
                    f"Auto-generate Dataset (current: {current_size}, target: 2000)"
                )
            else:
                print("⚠️ Auto-generation skipped: dependencies or API key not available")
                success_gen = False
        except:
            print("⚠️ Auto-generation skipped: dependency check failed")
            success_gen = False
    else:
        print(f"Dataset size adequate ({current_size} entries), proceeding with audit...")

    # Step 1: Rebalance dataset
    success1 = run_command(
        "python rebalance_dataset.py",
        "Dataset Rebalancing & Quality Enhancement"
    )

    # Step 2: Run comprehensive audit and cleanup
    success2 = run_command(
        "python audit_and_cleanup.py",
        "Comprehensive Dataset Audit & Cleanup"
    )

    # Step 3: Validate all datasets
    success3 = run_command(
        "python scripts/validate_judgmental_dataset.py",
        "Dataset Structure & Quality Validation"
    )

    # Step 4: Convert to Alpaca format
    success4 = run_command(
        "python convert_judgmental_to_alpaca.py",
        "Convert to Alpaca Format"
    )

    # Step 5: Create human review sample
    success5 = run_command(
        "python scripts/create_human_review_sample.py",
        "Create Human Review Sample"
    )

    # Summary
    print(f"\n{'='*50}")
    print("AUDIT & CLEANUP SUMMARY")
    print(f"{'='*50}")

    steps = [
        ("Dataset Rebalancing", success1),
        ("Dataset Audit & Cleanup", success2),
        ("Structure Validation", success3),
        ("Alpaca Format Conversion", success4),
        ("Human Review Sample", success5)
    ]

    all_success = True
    for step_name, success in steps:
        status = "✅ COMPLETED" if success else "❌ FAILED"
        print(f"{step_name}: {status}")
        if not success:
            all_success = False

    print(f"\n{'='*50}")
    if all_success:
        print("🎉 ALL AUDIT STEPS COMPLETED SUCCESSFULLY!")
        print("\nYour project is now:")
        print("✅ Validated and cleaned")
        print("✅ Ready for training")
        print("✅ Human review sample prepared")
        print("\nNext steps:")
        print("1. Review the files in review/ directory")
        print("2. Check logs/ for detailed validation reports")
        print("3. Use data/ files for model training")
    else:
        print("⚠️  SOME STEPS FAILED - CHECK LOGS ABOVE")
        print("Review error messages and retry failed steps.")

    print(f"{'='*50}")

if __name__ == "__main__":
    main()