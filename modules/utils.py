import logging
import os
import json
from datetime import datetime
from typing import Any, Dict

def create_output_directories():
    """Create necessary output directories"""
    directories = ["output", "intermediate", "logs", "raw", "data"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def validate_environment() -> Dict[str, str]:
    """Validate required environment variables"""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    
    return {"gemini_api_key": gemini_api_key}

def save_intermediate_file(data: Any, filename: str, file_type: str = "text") -> None:
    """Save intermediate files for debugging and pipeline recovery."""
    os.makedirs("intermediate", exist_ok=True)
    filepath = os.path.join("intermediate", filename)
    
    if file_type == "text":
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(data)
    elif file_type == "json":
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    elif file_type == "jsonl":
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

def setup_logging(log_file: str = "process.log") -> logging.Logger:
    """Set up comprehensive logging for the pipeline."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", log_file)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"Starting bilingual Q&A generation pipeline at {datetime.now()}")
    logger.info("=" * 80)
    
    return logger

def save_intermediate_file(data: Any, filename: str, file_type: str = "text") -> None:
    """Save intermediate files for debugging and review."""
    # Create intermediate directory if it doesn't exist
    os.makedirs("intermediate", exist_ok=True)
    filepath = os.path.join("intermediate", filename)
    
    logger = logging.getLogger(__name__)
    
    try:
        if file_type == "text":
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(data)
        elif file_type == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif file_type == "jsonl":
            with open(filepath, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved intermediate file: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save intermediate file {filepath}: {e}")

def validate_environment() -> Dict[str, str]:
    """Validate that all required environment variables are present."""
    logger = logging.getLogger(__name__)
    
    # Get API key from environment or use provided fallback
    api_key = os.getenv("GEMINI_API_KEY", "AIzaSyBbidR_bEfiMrhOufE4PAHrYEBvuPuqakg")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    
    logger.info("Environment validation successful")
    return {"gemini_api_key": api_key}

def create_output_directories() -> None:
    """Create necessary output directories."""
    directories = ["logs", "intermediate", "output"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def calculate_progress(current: int, total: int) -> str:
    """Calculate and format progress percentage."""
    if total == 0:
        return "0%"
    percentage = (current / total) * 100
    return f"{percentage:.1f}%"
