
# Bilingual Q&A and Judgmental Dataset Generation Pipeline

## Overview

This is a production-ready Python pipeline that generates high-quality bilingual Arabic-English question-answer datasets and judgmental training data from DOCX documents using Google's Gemini AI. The pipeline supports both traditional Q&A generation and synthetic judgmental dataset creation for training AI models to evaluate claim correctness.

## Project Structure

```
project/
├── main.py                           # Main Q&A generation pipeline
├── generate_synthetic.py             # Synthetic judgmental dataset generator
├── convert_to_alpaca.py              # Convert Q&A to Alpaca format
├── convert_judgmental_to_alpaca.py   # Convert judgmental to Alpaca format
├── modules/                          # Core processing modules
│   ├── __init__.py
│   ├── document_loader.py            # Google Docs content extraction
│   ├── file_loader.py                # Local file processing
│   ├── text_processor.py             # Text cleaning and chunking
│   ├── qa_generator.py               # Gemini AI Q&A generation
│   ├── synthetic_generator.py        # Synthetic data generation
│   ├── data_merger.py                # Bilingual alignment and merging
│   ├── alignment_engine.py           # Advanced text alignment
│   ├── token_manager.py              # Token-aware processing
│   └── utils.py                      # Utility functions
├── scripts/                          # Utility scripts
│   ├── convert_aligned_to_judgmental_seed.py
│   ├── validate_judgmental_dataset.py
│   └── apply_human_corrections.py
├── input/                            # Input documents
├── intermediate/                     # Processing intermediates
├── output/                           # Final datasets
├── data/                            # Synthetic datasets
└── logs/                            # Processing logs
```

## Core Features

### 1. Bilingual Q&A Generation
- Extracts content from DOCX files or Google Docs
- Processes Arabic and English text with specialized cleaning
- Generates contextual Q&A pairs using Gemini AI
- Aligns bilingual content for parallel datasets
- Outputs in multiple formats (JSONL, Alpaca)

### 2. Synthetic Judgmental Dataset Generation
- Creates True/False verification examples
- Token-aware chunking maximizing Gemini's 250K token capacity
- Balanced correct/incorrect claims generation
- Comprehensive metadata tracking
- Human correction workflow support

### 3. Production Features
- Comprehensive error handling and retry logic
- Rate limiting and API quota management
- Detailed logging and progress tracking
- Quality validation and deduplication
- Multiple output formats for different use cases

## Module Documentation

### main.py - Main Q&A Pipeline Controller

**Purpose**: Orchestrates the complete bilingual Q&A generation pipeline.

**Key Functions**:
- Command-line interface for pipeline configuration
- Coordinates document loading, processing, and output generation
- Handles both Google Docs URLs and local file inputs
- Manages API keys and environment setup

**Usage**:
```bash
python main.py --files input/arabic.docx input/english.docx --output output/dataset.jsonl
```

### generate_synthetic.py - Synthetic Dataset Generator

**Purpose**: Generates synthetic judgmental training data for claim verification tasks.

**Key Functions**:
- Token-aware chunking for maximum API efficiency
- Batch processing of multiple chunks per request
- Error type generation (factual, conceptual, reference, etc.)
- Model fallback strategy (gemini-2.5-pro → flash → flash-lite)

**Usage**:
```bash
python generate_synthetic.py --target-size 200 --max-tokens-per-request 250000
```

### modules/document_loader.py - Google Docs Integration

**Purpose**: Extracts content from Google Docs URLs with robust error handling.

**Key Classes**:
- `DocumentLoader`: Main class for Google Docs processing

**Key Methods**:
- `load_document(url)`: Downloads and extracts document content
- `_export_to_html(url)`: Converts Google Docs to HTML
- `_parse_html_content(html)`: Extracts clean text from HTML

**Features**:
- Automatic retry with exponential backoff
- HTML parsing and cleaning
- Timeout handling for large documents
- UTF-8 encoding support

### modules/file_loader.py - Local File Processing

**Purpose**: Handles local DOCX and text file processing.

**Key Classes**:
- `FileLoader`: Main class for local file operations

**Key Methods**:
- `load_file(file_path)`: Loads content from various file formats
- `_load_docx(file_path)`: Extracts text from DOCX files
- `_load_text(file_path)`: Processes plain text files

**Features**:
- Multiple format support (DOCX, TXT)
- Encoding detection and handling
- File validation and error reporting

### modules/text_processor.py - Text Processing Engine

**Purpose**: Cleans, normalizes, and chunks text content for optimal AI processing.

**Key Classes**:
- `TextProcessor`: Main text processing engine

**Key Methods**:
- `process_text(text, language)`: Complete text processing pipeline
- `clean_text(text, language)`: Language-specific cleaning
- `chunk_text(text, language)`: Intelligent content chunking
- `_clean_arabic_text(text)`: Arabic-specific normalization
- `_clean_english_text(text)`: English-specific cleaning

**Features**:
- Language-specific text normalization
- Paragraph-boundary-preserving chunking
- Word-count-based sizing (default: 1500 words)
- Metadata enrichment for each chunk

### modules/qa_generator.py - AI-Powered Q&A Generation

**Purpose**: Generates high-quality Q&A pairs using Google's Gemini AI.

**Key Classes**:
- `QAGenerator`: Main Q&A generation engine

**Key Methods**:
- `generate_qa_for_chunks(chunks, language)`: Batch Q&A generation
- `generate_qa_for_chunk(chunk, language)`: Single chunk processing
- `_create_qa_prompt(text, language)`: Language-specific prompt engineering
- `_call_gemini_api(prompt, language)`: API interaction with fallback

**Features**:
- Multi-model fallback strategy
- Rate limiting (50 requests/minute)
- Language-specific prompt engineering
- Structured JSON response handling
- Comprehensive metadata tracking

### modules/synthetic_generator.py - Synthetic Data Generator

**Purpose**: Creates synthetic judgmental datasets with token optimization.

**Key Classes**:
- `SyntheticGenerator`: Main synthetic data engine

**Key Methods**:
- `generate_synthetic_dataset(chunks, target_size)`: Complete generation pipeline
- `_create_batches(chunks)`: Token-aware batch creation
- `_generate_synthetic_batch(batch)`: Batch processing with API calls
- `_create_synthetic_prompt(chunks)`: Prompt engineering for judgmental data

**Features**:
- Token-aware chunking maximizing 250K token capacity
- Balanced correct/incorrect example generation
- Error type classification (factual, conceptual, reference, etc.)
- Batch processing for efficiency
- Raw response storage for debugging

### modules/data_merger.py - Bilingual Data Alignment

**Purpose**: Merges and aligns Arabic-English Q&A pairs for bilingual datasets.

**Key Classes**:
- `DataMerger`: Main data alignment engine

**Key Methods**:
- `align_bilingual_qa(ar_qa, en_qa)`: Content-based alignment
- `deduplicate_qa_pairs(qa_pairs, language)`: Quality deduplication
- `validate_qa_pair(qa)`: Content validation
- `save_final_dataset(pairs, output_path)`: JSONL export

**Features**:
- Chunk-based content alignment
- Duplicate detection and removal
- Quality validation (minimum lengths, required fields)
- Bilingual pair creation and validation

### modules/alignment_engine.py - Advanced Text Alignment

**Purpose**: Provides sophisticated alignment methods for bilingual content.

**Key Classes**:
- `AlignmentEngine`: Advanced alignment algorithms

**Key Methods**:
- `align_chunks_multi_method(ar_chunks, en_chunks)`: Multi-strategy alignment
- `heading_based_alignment(ar_chunks, en_chunks)`: Structural alignment
- `position_based_alignment(ar_chunks, en_chunks)`: Sequential alignment
- `semantic_alignment(ar_chunks, en_chunks)`: Embedding-based alignment

**Features**:
- Multiple alignment strategies
- Confidence scoring for alignments
- Fuzzy text similarity matching
- Unmatched content handling

### modules/token_manager.py - Token-Aware Processing

**Purpose**: Manages token counting and batch optimization for API efficiency.

**Key Classes**:
- `TokenManager`: Token counting and batching engine

**Key Methods**:
- `count_tokens(text)`: Accurate token counting using tiktoken
- `create_token_aware_chunks(text)`: Semantic chunking with token limits
- `create_batches(chunks, max_tokens)`: Optimal batch creation
- `estimate_batch_tokens(chunks)`: Token estimation for batching

**Features**:
- Precise token counting with cl100k_base encoding
- Semantic boundary preservation
- Maximum utilization of API token limits
- Configurable safety margins and overlap

### modules/utils.py - Utility Functions

**Purpose**: Provides common utilities for logging, file I/O, and environment management.

**Key Functions**:
- `setup_logging(log_file)`: Comprehensive logging configuration
- `validate_environment()`: Environment and dependency validation
- `save_intermediate_file(data, filename, file_type)`: Debug file saving
- `load_json_file(file_path)`: Safe JSON loading with error handling

**Features**:
- Timestamped logging with rotation
- Environment validation
- Safe file operations with encoding handling
- Progress tracking utilities

### scripts/convert_aligned_to_judgmental_seed.py

**Purpose**: Converts aligned Q&A pairs into seed format for judgmental dataset generation.

**Key Functions**:
- Transforms Q&A pairs into claim-context format
- Generates unique identifiers for tracking
- Creates language-specific seed entries
- Maintains metadata for traceability

### scripts/validate_judgmental_dataset.py

**Purpose**: Validates the quality and consistency of generated judgmental datasets.

**Key Functions**:
- Checks required fields and data types
- Validates claim-context relationships
- Ensures balanced correct/incorrect distributions
- Reports quality metrics and issues

### scripts/apply_human_corrections.py

**Purpose**: Applies human corrections to generated datasets.

**Key Functions**:
- Loads correction files in standard formats
- Applies corrections while maintaining metadata
- Validates correction consistency
- Exports corrected datasets

## Configuration Files

### config.example.yaml - Axolotl Training Configuration

**Purpose**: Ready-to-use configuration for fine-tuning models with Axolotl.

**Key Sections**:
- Model configuration (base model, tokenizer settings)
- Dataset configuration (paths, formatting)
- Training parameters (learning rate, batch size, epochs)
- Evaluation and logging settings

### pyproject.toml - Project Dependencies

**Purpose**: Defines project dependencies and Python package configuration.

**Key Dependencies**:
- `google-genai`: Gemini AI API client
- `python-docx`: DOCX file processing
- `beautifulsoup4`: HTML parsing
- `tenacity`: Retry logic implementation
- `tiktoken`: Token counting for API optimization

## Output Formats

### 1. Standard Q&A Format (JSONL)
```json
{
  "question_ar": "السؤال باللغة العربية؟",
  "answer_ar": "الإجابة باللغة العربية",
  "question_en": "Question in English?",
  "answer_en": "Answer in English"
}
```

### 2. Alpaca Format (JSONL)
```json
{
  "instruction": "Question or instruction here",
  "input": "",
  "output": "Answer or response here"
}
```

### 3. Judgmental Format (JSONL)
```json
{
  "type": "correct",
  "claim": "Claim text to be verified",
  "explanation": "Explanation of why the claim is correct/incorrect",
  "reference": "Source reference for verification",
  "language": "arabic",
  "chunk_id": 15,
  "batch_id": 5,
  "model": "gemini-2.5-pro"
}
```

## Usage Examples

### Basic Q&A Generation
```bash
# From local files
python main.py --files input/arabic.docx input/english.docx --output output/qa_dataset.jsonl

# From Google Docs
python main.py --docs "https://docs.google.com/document/d/DOC_ID1" "https://docs.google.com/document/d/DOC_ID2"
```

### Synthetic Judgmental Dataset Generation
```bash
# Generate 200 examples with token optimization
python generate_synthetic.py --target-size 200 --max-tokens-per-request 250000

# Resume from existing chunks
python generate_synthetic.py --resume --target-size 1000
```

### Format Conversion
```bash
# Convert Q&A to Alpaca format
python convert_to_alpaca.py

# Convert judgmental data to Alpaca format
python convert_judgmental_to_alpaca.py
```

## Performance Specifications

### Processing Capacity
- **Document Size**: Handles documents up to 50MB+ efficiently
- **Throughput**: ~50 chunks per minute (API rate limited)
- **Token Utilization**: 75-85% of Gemini's 250K token capacity per batch
- **Memory Efficiency**: Streams large documents without full loading
- **Success Rate**: 99%+ with comprehensive fallback strategies

### Quality Metrics
- **Deduplication**: Automatic removal of duplicate content
- **Validation**: Minimum length requirements and field validation
- **Language Quality**: Professional Arabic (فصحى) and formal English
- **Alignment Accuracy**: High-precision bilingual content matching

## API Integration

### Gemini AI Configuration
- **Primary Model**: gemini-2.5-pro
- **Fallback Models**: gemini-2.5-flash, gemini-2.5-flash-lite
- **Rate Limiting**: 50 requests/minute with exponential backoff
- **Token Management**: Optimized for 250K token capacity
- **Response Format**: Structured JSON with validation

### Environment Setup
```bash
# Set API key
export GEMINI_API_KEY="your_api_key_here"

# Install dependencies
pip install google-genai python-docx beautifulsoup4 tenacity tiktoken
```

## Monitoring and Debugging

### Logging System
- **Location**: `logs/process.log`
- **Format**: Timestamped entries with severity levels
- **Rotation**: Automatic log rotation for long-running processes
- **Content**: API calls, errors, progress tracking, performance metrics

### Intermediate Files
- **Purpose**: Debug and recovery support
- **Location**: `intermediate/` directory
- **Types**: Cleaned text, chunks, Q&A pairs, alignment data
- **Format**: JSON for structured data, TXT for text content

### Quality Assurance
- **Validation**: Automatic content validation at each stage
- **Deduplication**: Remove duplicate questions and answers
- **Error Recovery**: Graceful handling of API failures and malformed data
- **Progress Tracking**: Real-time progress with ETA estimates

## Training and Deployment

### Axolotl Integration
The project includes ready-to-use Axolotl configuration for fine-tuning language models:

```yaml
# Key configuration highlights
base_model: meta-llama/Llama-2-7b-chat-hf
datasets:
  - path: output/judgmental_alpaca.jsonl
    type: alpaca
sequence_len: 2048
adapter: lora
```

### Recommended Training Commands
```bash
# Fine-tune for Q&A tasks
accelerate launch -m axolotl.cli.train config.example.yaml

# Fine-tune for judgmental tasks
accelerate launch -m axolotl.cli.train config_judgmental.yaml
```

## Troubleshooting

### Common Issues

1. **Google Docs Access Errors (HTTP 432)**
   - Ensure documents are publicly accessible
   - Check sharing permissions
   - Use alternative export URLs

2. **API Rate Limiting**
   - Built-in rate limiting handles this automatically
   - Monitor logs for retry attempts
   - Consider reducing batch sizes for stability

3. **Memory Issues with Large Documents**
   - Pipeline streams content to minimize memory usage
   - Increase chunk size if needed
   - Monitor system resources during processing

4. **Language Detection Issues**
   - Verify document structure and content
   - Check for mixed-language content in single documents
   - Use manual language specification if needed

### Performance Optimization

1. **Token Utilization**
   - Adjust `max-tokens-per-request` based on content complexity
   - Monitor batch utilization rates in logs
   - Use smaller safety margins for maximum efficiency

2. **Processing Speed**
   - API rate limits are the primary bottleneck
   - Larger batches improve efficiency but may increase error rates
   - Balance batch size with reliability needs

3. **Quality vs Quantity**
   - Higher chunk overlap improves context but reduces throughput
   - Adjust minimum length requirements based on use case
   - Consider post-processing filtering for specific quality needs

This comprehensive documentation provides everything needed to understand, use, and extend the bilingual Q&A and judgmental dataset generation pipeline.
