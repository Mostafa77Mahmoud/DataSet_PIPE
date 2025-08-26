#!/usr/bin/env python3
"""
Bilingual Q&A Dataset Generation Pipeline

This script generates a high-quality bilingual Arabic-English Q&A dataset
from Google Docs using Gemini AI.
"""

import argparse
import os
import sys
from typing import Optional

# Add modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.utils import setup_logging, validate_environment, create_output_directories
from modules.document_loader import GoogleDocsLoader
from modules.file_loader import FileLoader
from modules.text_processor import TextProcessor
from modules.qa_generator import QAGenerator
from modules.data_merger import DataMerger

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate bilingual Q&A dataset from Google Docs using Gemini AI"
    )
    
    # Document source arguments (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    
    source_group.add_argument(
        "--docs",
        nargs=2,
        metavar=("AR_DOC_URL", "EN_DOC_URL"),
        help="Arabic and English Google Docs URLs"
    )
    
    source_group.add_argument(
        "--files",
        nargs=2,
        metavar=("AR_FILE", "EN_FILE"),
        default=["input/Shariaah-Standards-ARB-compressed.docx", "input/Shariaa-Standards-ENG.docx"],
        help="Arabic and English local file paths (default: input/Shariaah-Standards-ARB-compressed.docx input/Shariaa-Standards-ENG.docx)"
    )
    
    # Legacy support for original argument names
    parser.add_argument(
        "--ar-doc",
        help="Arabic Google Docs URL (legacy)"
    )
    
    parser.add_argument(
        "--en-doc", 
        help="English Google Docs URL (legacy)"
    )
    
    parser.add_argument(
        "--api-key",
        help="Gemini API key (can also be set via GEMINI_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1500,
        help="Target chunk size in words (default: 1500)"
    )
    
    parser.add_argument(
        "--output",
        default="output/final_bilingual_qa.jsonl",
        help="Output JSONL file path (default: output/final_bilingual_qa.jsonl)"
    )
    
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro",
        help="Gemini model version (default: gemini-2.5-pro)"
    )
    
    return parser.parse_args()

def main():
    """Main pipeline execution."""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directories
    create_output_directories()
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Validate environment and get API key
        if args.api_key:
            os.environ["GEMINI_API_KEY"] = args.api_key
        
        env_vars = validate_environment()
        api_key = env_vars["gemini_api_key"]
        
        logger.info("Starting bilingual Q&A generation pipeline")
        
        # Log input sources
        if args.docs:
            logger.info(f"Arabic document: {args.docs[0]}")
            logger.info(f"English document: {args.docs[1]}")
        elif args.files:
            logger.info(f"Arabic file: {args.files[0]}")
            logger.info(f"English file: {args.files[1]}")
        elif args.ar_doc and args.en_doc:
            logger.info(f"Arabic document: {args.ar_doc}")
            logger.info(f"English document: {args.en_doc}")
        
        logger.info(f"Chunk size: {args.chunk_size} words")
        logger.info(f"Output file: {args.output}")
        logger.info(f"Model: {args.model}")
        
        # Step 1: Load documents (Google Docs or local files)
        logger.info("=" * 50)
        logger.info("STEP 1: Loading Documents")
        logger.info("=" * 50)
        
        # Determine input source and load documents
        if args.docs:
            ar_url, en_url = args.docs
            loader = GoogleDocsLoader()
            ar_text, en_text = loader.load_documents(ar_url, en_url)
        elif args.files:
            ar_file, en_file = args.files
            loader = FileLoader()
            ar_text, en_text = loader.load_local_files(ar_file, en_file)
        elif args.ar_doc and args.en_doc:  # Legacy support
            loader = GoogleDocsLoader()
            ar_text, en_text = loader.load_documents(args.ar_doc, args.en_doc)
        else:
            raise ValueError("Must provide either --docs URLs or --files paths")
        
        # Create combined AAOIFI content for judgmental dataset generation
        os.makedirs("intermediate", exist_ok=True)
        combined_content = f"=== ARABIC CONTENT ===\n{ar_text}\n\n=== ENGLISH CONTENT ===\n{en_text}"
        with open("intermediate/aaofi_cleaned.txt", 'w', encoding='utf-8') as f:
            f.write(combined_content)
        logger.info(f"Created intermediate/aaofi_cleaned.txt with {len(combined_content)} characters")
        
        # Step 2: Preprocess and chunk documents
        logger.info("=" * 50)
        logger.info("STEP 2: Preprocessing and chunking")
        logger.info("=" * 50)
        
        processor = TextProcessor(chunk_size=args.chunk_size)
        aligned_chunks = processor.process_documents(ar_text, en_text)
        
        # Step 3: Generate Q&A pairs using Gemini
        logger.info("=" * 50)
        logger.info("STEP 3: Generating Q&A pairs with Gemini")
        logger.info("=" * 50)
        
        qa_generator = QAGenerator(api_key=api_key, model=args.model)
        ar_qa_pairs, en_qa_pairs = qa_generator.generate_bilingual_qa(aligned_chunks)
        
        # Step 4: Merge and align bilingual Q&A
        logger.info("=" * 50)
        logger.info("STEP 4: Merging bilingual Q&A pairs")
        logger.info("=" * 50)
        
        merger = DataMerger()
        merger.merge_and_save(ar_qa_pairs, en_qa_pairs, args.output)
        
        # Step 5: Final summary
        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
        
        logger.info(f"Final dataset saved to: {args.output}")
        
        # Print file size and count
        if os.path.exists(args.output):
            file_size = os.path.getsize(args.output)
            with open(args.output, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            
            logger.info(f"Dataset contains {line_count} Q&A pairs")
            logger.info(f"File size: {file_size:,} bytes")
        
        logger.info("Check the 'intermediate' directory for debugging files")
        logger.info("Check the 'logs' directory for detailed logs")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.exception("Full error details:")
        sys.exit(1)

if __name__ == "__main__":
    main()
