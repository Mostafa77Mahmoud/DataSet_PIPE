
"""
Hardened synthetic data generator with anti-fabrication controls.
"""

import json
import os
import logging
import re
from typing import Dict, List, Optional
from difflib import SequenceMatcher

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logging.warning("google.generativeai not available")

logger = logging.getLogger(__name__)

HARDENED_PROMPT_EN = """You are an expert Islamic finance validator. Based ONLY on the provided AAOIFI content, create a judgment task.

STRICT RULES:
1. NEVER invent or fabricate references
2. If you cannot find the exact reference in the provided content, use "UNKNOWN"
3. Only use temperature=0.0 for consistency
4. Base claims ONLY on the provided AAOIFI text

Provided AAOIFI Content:
{context}

Create a JSON response with this EXACT format:
{{
  "instruction": "Evaluate this Islamic finance claim for compliance with AAOIFI standards:",
  "input": "[Clear, specific claim about Islamic finance]",
  "output": "VERDICT: [TRUE/FALSE]\\n\\nExplanation: [Clear reasoning based on AAOIFI standards]\\n\\nReference: [Exact AAOIFI reference or UNKNOWN]",
  "meta": {{
    "verdict": "[TRUE/FALSE]",
    "language": "english",
    "reference_verified": true,
    "suspected_fabrication": false
  }}
}}

Respond with valid JSON only."""

HARDENED_PROMPT_AR = """أنت خبير في التمويل الإسلامي. بناءً فقط على محتوى هيئة المحاسبة والمراجعة المقدم، أنشئ مهمة تقييم.

قواعد صارمة:
1. لا تخترع أو تفبرك المراجع أبداً
2. إذا لم تجد المرجع الدقيق في المحتوى المقدم، استخدم "غير معروف"
3. استخدم فقط temperature=0.0 للثبات
4. بني الادعاءات فقط على النص المقدم من هيئة المحاسبة

محتوى هيئة المحاسبة المقدم:
{context}

أنشئ استجابة JSON بهذا التنسيق الدقيق:
{{
  "instruction": "قم بتقييم هذا الادعاء في التمويل الإسلامي للامتثال لمعايير هيئة المحاسبة:",
  "input": "[ادعاء واضح ومحدد حول التمويل الإسلامي]",
  "output": "الحكم: [صحيح/خاطئ]\\n\\nالتفسير: [تبرير واضح بناءً على معايير هيئة المحاسبة]\\n\\nالمرجع: [مرجع دقيق من هيئة المحاسبة أو غير معروف]",
  "meta": {{
    "verdict": "[صحيح/خاطئ]",
    "language": "arabic", 
    "reference_verified": true,
    "suspected_fabrication": false
  }}
}}

استجب بـ JSON صالح فقط."""

def similarity_score(a: str, b: str) -> float:
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def verify_reference_against_content(reference: str, content: str, threshold: float = 0.85) -> bool:
    """Verify if reference exists in content with fuzzy matching."""
    if not reference or reference.upper() == "UNKNOWN":
        return True
    
    # Check exact match first
    if reference.lower() in content.lower():
        return True
    
    # Check fuzzy match
    sentences = content.split('.')
    for sentence in sentences:
        if similarity_score(reference, sentence.strip()) >= threshold:
            return True
    
    return False

def load_aaoifi_content() -> str:
    """Load combined AAOIFI content for reference verification."""
    content_file = "intermediate/aaofi_cleaned.txt"
    if os.path.exists(content_file):
        with open(content_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def generate_batch(context: str, count: int = 10, language: str = "english") -> List[Dict]:
    """Generate a batch of synthetic examples with anti-fabrication controls."""
    if not GENAI_AVAILABLE:
        logger.error("google.generativeai not available")
        return []
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.error("GEMINI_API_KEY not found")
        return []
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = HARDENED_PROMPT_AR if language == "arabic" else HARDENED_PROMPT_EN
        aaoifi_content = load_aaoifi_content()
        
        results = []
        for i in range(count):
            try:
                response = model.generate_content(
                    prompt.format(context=context[:4000]),  # Limit context size
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.0,
                        top_p=0.95,
                        top_k=40,
                        max_output_tokens=1000
                    )
                )
                
                # Parse response
                content = response.text.strip()
                if content.startswith('```json'):
                    content = content[7:-3]
                elif content.startswith('```'):
                    content = content[3:-3]
                
                data = json.loads(content)
                
                # Extract reference for verification
                output_text = data.get('output', '')
                reference_match = re.search(r'Reference: (.+?)(?:\n|$)', output_text)
                reference = reference_match.group(1).strip() if reference_match else "UNKNOWN"
                
                # Verify reference against AAOIFI content
                is_verified = verify_reference_against_content(reference, aaoifi_content)
                
                if not is_verified and reference.upper() != "UNKNOWN":
                    # Mark as suspected fabrication and set to UNKNOWN
                    data['meta']['suspected_fabrication'] = True
                    data['meta']['reference_verified'] = False
                    # Update output to use UNKNOWN
                    data['output'] = re.sub(r'Reference: .+?(?=\n|$)', 'Reference: UNKNOWN', data['output'])
                    logger.warning(f"Suspected fabricated reference replaced: {reference}")
                else:
                    data['meta']['reference_verified'] = True
                    data['meta']['suspected_fabrication'] = False
                
                results.append(data)
                
            except Exception as e:
                logger.error(f"Error generating example {i}: {e}")
                continue
        
        logger.info(f"Generated {len(results)} examples in {language}")
        return results
        
    except Exception as e:
        logger.error(f"Error in batch generation: {e}")
        return []

def post_process_batch(examples: List[Dict]) -> List[Dict]:
    """Post-process batch to remove duplicates and normalize."""
    seen_inputs = set()
    processed = []
    
    for example in examples:
        input_text = example.get('input', '').strip().lower()
        if input_text not in seen_inputs:
            seen_inputs.add(input_text)
            processed.append(example)
    
    logger.info(f"Post-processed: {len(processed)} unique examples from {len(examples)}")
    return processed
