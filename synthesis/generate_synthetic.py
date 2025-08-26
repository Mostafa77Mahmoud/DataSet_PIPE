#!/usr/bin/env python3
"""
Synthetic data generation using the new comprehensive dataset generator.
"""

import sys
import os
import argparse

# Add the parent directory to the path so we can import the main generator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_judgmental_dataset import DatasetGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic AAOIFI judgmental dataset")
    parser.add_argument("--target-size", type=int, default=3000, help="Target dataset size")
    parser.add_argument("--concurrency", type=int, default=2, help="API concurrency level")
    parser.add_argument("--preview", action="store_true", help="Generate small preview")

    args = parser.parse_args()

    if args.preview:
        target_size = 50
        print("🔍 PREVIEW MODE: Generating 50 examples")
    else:
        target_size = args.target_size

    # Use high-availability API keys
    api_keys = [
        "AIzaSyBbidR_bEfiMrhOufE4PAHrYEBvuPuqakg",
        "AIzaSyAIPk1An1O6sZiro64Q4R9PjVrqvPkSVvQ"
    ]
    
    # Set environment variable for compatibility
    os.environ['GEMINI_API_KEY'] = api_keys[0]
    
    generator = DatasetGenerator(
        target_size=target_size,
        concurrency=args.concurrency
    )

    success = generator.generate_dataset()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())