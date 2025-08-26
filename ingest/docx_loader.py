
"""
DOCX document loader for AAOIFI standards.
"""

import os
from pathlib import Path
from docx import Document
import logging

logger = logging.getLogger(__name__)

def load_docx_content(file_path: str) -> str:
    """Load and extract text content from DOCX file."""
    try:
        doc = Document(file_path)
        content = []
        
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                content.append(text)
        
        return '\n'.join(content)
    
    except Exception as e:
        logger.error(f"Error loading DOCX file {file_path}: {e}")
        return ""

def process_aaoifi_documents(arabic_file: str, english_file: str, output_dir: str = "intermediate"):
    """Process both Arabic and English AAOIFI documents."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Arabic content
    arabic_content = load_docx_content(arabic_file)
    arabic_output = os.path.join(output_dir, "arabic_cleaned.txt")
    with open(arabic_output, 'w', encoding='utf-8') as f:
        f.write(arabic_content)
    
    # Load English content  
    english_content = load_docx_content(english_file)
    english_output = os.path.join(output_dir, "english_cleaned.txt")
    with open(english_output, 'w', encoding='utf-8') as f:
        f.write(english_content)
    
    # Combine for reference checking
    combined_content = f"{arabic_content}\n\n{english_content}"
    combined_output = os.path.join(output_dir, "aaofi_cleaned.txt")
    with open(combined_output, 'w', encoding='utf-8') as f:
        f.write(combined_content)
    
    logger.info(f"Processed documents - Arabic: {len(arabic_content)} chars, English: {len(english_content)} chars")
    return arabic_output, english_output, combined_output
