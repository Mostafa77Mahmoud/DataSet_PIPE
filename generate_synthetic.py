
#!/usr/bin/env python3
"""
Synthetic Judgmental Dataset Generation CLI

Generate synthetic verification examples from Q&A seed data using Gemini API.
"""

import argparse
import os
import sys
import json
from typing import List, Dict

# Add modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.utils import setup_logging, validate_environment, create_output_directories
from modules.synthetic_generator import SyntheticGenerator, TokenConfig

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic judgmental dataset from Q&A seed data"
    )
    
    parser.add_argument(
        "--seed-file",
        default="data/judgmental_seed.jsonl",
        help="Input seed file path"
    )
    
    parser.add_argument(
        "--target-size",
        type=int,
        default=200,
        help="Target number of examples to generate"
    )
    
    parser.add_argument(
        "--models",
        default="gemini-2.5-pro,gemini-2.5-flash,gemini-2.5-flash-lite",
        help="Comma-separated list of models to use"
    )
    
    parser.add_argument(
        "--max-tokens-per-request",
        type=int,
        default=250000,
        help="Maximum tokens per API request"
    )
    
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.96,
        help="Safety margin for token limit"
    )
    
    parser.add_argument(
        "--per-chunk-output-tokens",
        type=int,
        default=2048,
        help="Expected output tokens per chunk"
    )
    
    parser.add_argument(
        "--paraphrase-temp",
        type=float,
        default=0.2,
        help="Temperature for paraphrase generation"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume generation from existing data"
    )
    
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Generate preview dataset and stop"
    )
    
    parser.add_argument(
        "--api-key",
        help="Gemini API key (can also be set via GEMINI_API_KEY environment variable)"
    )
    
    # Legacy arguments for compatibility
    parser.add_argument("--ar-file", help=argparse.SUPPRESS)
    parser.add_argument("--en-file", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--batching-mode", help=argparse.SUPPRESS)
    parser.add_argument("--chunk-overlap-sentences", help=argparse.SUPPRESS)
    parser.add_argument("--max-paragraph-tokens", help=argparse.SUPPRESS)
    
    return parser.parse_args()

def create_data_splits(examples: List[Dict], output_dir: str = "data") -> None:
    """Create train/val/test splits."""
    import random
    
    # Shuffle examples
    random.seed(42)
    shuffled = examples.copy()
    random.shuffle(shuffled)
    
    # Split ratios: 80% train, 10% val, 10% test
    total = len(shuffled)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    
    train_data = shuffled[:train_end]
    val_data = shuffled[train_end:val_end]
    test_data = shuffled[val_end:]
    
    # Save splits
    splits = {
        "train.jsonl": train_data,
        "val.jsonl": val_data,
        "test.jsonl": test_data
    }
    
    for filename, data in splits.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        print(f"Saved {len(data)} examples to {filepath}")

def create_metadata(examples: List[Dict], args: argparse.Namespace, output_dir: str = "data") -> None:
    """Create metadata file."""
    
    # Calculate statistics
    total = len(examples)
    ar_count = sum(1 for ex in examples if ex['meta']['language'] == 'ar')
    en_count = sum(1 for ex in examples if ex['meta']['language'] == 'en')
    correct_count = sum(1 for ex in examples if ex['meta']['type'] == 'correct')
    incorrect_count = sum(1 for ex in examples if ex['meta']['type'] == 'incorrect')
    
    metadata = {
        "total_examples": total,
        "language_distribution": {
            "arabic": ar_count,
            "english": en_count
        },
        "type_distribution": {
            "correct": correct_count,
            "incorrect": incorrect_count
        },
        "generation_config": {
            "target_size": args.target_size,
            "models": args.models.split(','),
            "max_tokens_per_request": args.max_tokens_per_request,
            "safety_margin": args.safety_margin,
            "per_chunk_output_tokens": args.per_chunk_output_tokens,
            "paraphrase_temp": args.paraphrase_temp
        },
        "splits": {
            "train": int(total * 0.8),
            "val": int(total * 0.1),
            "test": int(total * 0.1)
        }
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Saved metadata to {metadata_path}")

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Create output directories
    create_output_directories()
    for dir_name in ["data", "raw", "intermediate", "review"]:
        os.makedirs(dir_name, exist_ok=True)
    
    # Setup logging
    logger = setup_logging("synthetic_generation.log")
    
    try:
        # Validate environment and get API key
        if args.api_key:
            os.environ["GEMINI_API_KEY"] = args.api_key
        
        # Use new API key
        api_key = "AIzaSyBbidR_bEfiMrhOufE4PAHrYEBvuPuqakg"
        os.environ["GEMINI_API_KEY"] = api_key
        
        env_vars = validate_environment()
        api_key = env_vars["gemini_api_key"]
        
        # Check seed file exists
        if not os.path.exists(args.seed_file):
            logger.error(f"Seed file not found: {args.seed_file}")
            logger.info("Please run: python scripts/convert_aligned_to_judgmental_seed.py")
            sys.exit(1)
        
        logger.info("Starting synthetic judgmental dataset generation")
        logger.info(f"Seed file: {args.seed_file}")
        logger.info(f"Target size: {args.target_size}")
        logger.info(f"Models: {args.models}")
        logger.info(f"Preview mode: {args.preview}")
        
        # Configure token management
        token_config = TokenConfig(
            max_tokens_per_request=args.max_tokens_per_request,
            safety_margin=args.safety_margin,
            per_chunk_expected_output=args.per_chunk_output_tokens
        )
        
        # Set target size for preview
        target_size = 200 if args.preview else args.target_size
        
        # Initialize generator
        models = [m.strip() for m in args.models.split(',')]
        generator = SyntheticGenerator(
            api_key=api_key,
            models=models,
            token_config=token_config,
            paraphrase_temp=args.paraphrase_temp
        )
        
        # Generate dataset
        examples = generator.generate_synthetic_dataset(
            seed_file=args.seed_file,
            target_size=target_size,
            resume=args.resume
        )
        
        if not examples:
            logger.error("No examples generated")
            sys.exit(1)
        
        # Save main dataset
        if args.preview:
            output_file = "data/preview_first_200.jsonl"
        else:
            output_file = "data/judgmental_final.jsonl"
        
        generator.save_dataset(examples, output_file)
        
        # Create preview file if not in preview mode
        if not args.preview and len(examples) >= 200:
            preview_examples = examples[:200]
            generator.save_dataset(preview_examples, "data/preview_first_200.jsonl")
        
        # Create data splits (only for full dataset)
        if not args.preview:
            create_data_splits(examples)
        
        # Create metadata
        create_metadata(examples, args)
        
        # Print summary
        logger.info("=" * 50)
        logger.info("GENERATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
        logger.info(f"Generated {len(examples)} examples")
        logger.info(f"Saved to: {output_file}")
        
        ar_count = sum(1 for ex in examples if ex['meta']['language'] == 'ar')
        en_count = sum(1 for ex in examples if ex['meta']['language'] == 'en')
        correct_count = sum(1 for ex in examples if ex['meta']['type'] == 'correct')
        incorrect_count = sum(1 for ex in examples if ex['meta']['type'] == 'incorrect')
        
        logger.info(f"Language distribution: Arabic={ar_count}, English={en_count}")
        logger.info(f"Type distribution: Correct={correct_count}, Incorrect={incorrect_count}")
        
        if args.preview:
            logger.info("Preview mode completed. Run validation script next:")
            logger.info(f"python scripts/validate_judgmental_dataset.py --input {output_file}")
        
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        logger.exception("Full error details:")
        sys.exit(1)

if __name__ == "__main__":
    main()
