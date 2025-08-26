# Overview

This is a production-ready Python pipeline that generates high-quality bilingual Arabic-English question-answer datasets from Google Docs or local files using Google's Gemini AI. The system extracts content from multiple sources, processes and chunks the text while maintaining alignment between languages, uses Gemini 2.5-Pro to generate contextual Q&A pairs, and produces clean, deduplicated JSONL files ready for AI model fine-tuning.

## Recent Updates (August 12, 2025)
- **Enhanced Input Options**: Added support for both Google Docs URLs and local file inputs
- **DOCX File Support**: Added python-docx integration for processing Microsoft Word documents  
- **Improved Error Handling**: Better timeout management and access permission diagnostics  
- **Flexible CLI**: Updated command structure with `--docs` and `--files` options
- **Enhanced System Prompt**: Implemented powerful quality control with strict format adherence and consistency enforcement
- **Model Fallback Strategy**: Added automatic switching between gemini-2.5-pro → gemini-2.5-flash → gemini-2.5-flash-lite for maximum reliability
- **Quality Optimization**: Enhanced prompts requiring exactly 5 Q&A pairs per chunk with specialized instructions for both languages
- **Access Issue Resolution**: Created comprehensive solution guide for Google Docs permission issues

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Modular Pipeline Design
The system follows a modular architecture with distinct components for each stage of the pipeline:
- **Document Loading**: Handles Google Docs content extraction via HTML export
- **Text Processing**: Cleans, normalizes, and chunks text while preserving structure
- **Q&A Generation**: Interfaces with Gemini AI to create contextual question-answer pairs
- **Data Merging**: Aligns bilingual data and handles deduplication

## Text Processing Strategy
The pipeline implements intelligent text chunking that splits documents into ~1500-word segments while preserving paragraph boundaries. It maintains chunk alignment between Arabic and English documents to ensure corresponding sections are processed together. Language-specific preprocessing handles Arabic punctuation normalization and UTF-8 encoding requirements.

## AI Integration Pattern
The system uses Google's Gemini 2.5-Pro model with enhanced structured prompting and intelligent model fallback for maximum reliability. It implements:
- **Enhanced System Prompt**: Strict quality control with format adherence and consistency enforcement
- **Model Fallback Strategy**: Automatic switching between gemini-2.5-pro → gemini-2.5-flash → gemini-2.5-flash-lite
- **Rate Limiting**: 50 requests per minute with exponential backoff retry logic
- **Quality Optimization**: Specialized prompts requiring exactly 5 Q&A pairs per chunk
- **Language-Specific Processing**: Culturally appropriate and contextually relevant prompts for both Arabic and English

## Error Handling and Quality Assurance
The architecture includes comprehensive logging throughout the pipeline, intermediate file saving for debugging and review, and validation mechanisms to ensure data quality. Deduplication logic removes identical questions within each language while preserving content integrity.

## Command-Line Interface
The system provides a flexible CLI that accepts Google Docs URLs, API keys (via environment variable or command line), and configurable chunk sizes. The main entry point orchestrates the entire pipeline with proper error handling and progress reporting.

# External Dependencies

## AI Services
- **Google Gemini AI**: Uses the `google-genai` client library to interface with Gemini 2.5-Pro model for Q&A generation
- **Gemini API Key**: Required for authentication with Google's AI services

## Document Access
- **Google Docs Export API**: Accesses documents via HTML export functionality without requiring Google Drive API credentials
- **Local File Support**: Alternative file input system for when Google Docs access is restricted
- **HTTP Requests**: Uses the `requests` library for document downloading with session management and retry logic
- **Access Validation**: Built-in testing and fallback mechanisms for permission issues

## Text Processing Libraries
- **BeautifulSoup4**: Parses HTML content from Google Docs exports
- **python-docx**: Extracts text content from Microsoft Word DOCX files
- **Tenacity**: Provides retry mechanisms for API calls with configurable backoff strategies

## Python Standard Libraries
- **argparse**: Command-line argument parsing
- **logging**: Comprehensive logging throughout the pipeline
- **json**: Data serialization for intermediate files and final output
- **re**: Regular expression processing for text cleaning and normalization
- **os/sys**: File system operations and path management