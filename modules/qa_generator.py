import json
import logging
import time
from typing import List, Dict, Optional
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
from modules.utils import save_intermediate_file

class QAGenerator:
    """Class to generate Q&A pairs using Gemini API."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-pro"):
        self.api_key = api_key
        self.model = model
        self.client = genai.Client(api_key=api_key)
        self.logger = logging.getLogger(__name__)
        
        # Model fallback strategy
        self.available_models = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
        self.current_model_index = 0
        
        # Rate limiting
        self.requests_per_minute = 50
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Implement rate limiting for API calls."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        min_interval = 60.0 / self.requests_per_minute
        
        if time_since_last_request < min_interval:
            sleep_time = min_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_enhanced_system_instruction(self) -> str:
        """Get enhanced system instruction for consistent Q&A generation."""
        return """You are the **controller** of a bilingual (Arabic-English) Q&A generation pipeline.
Your role is to **strictly enforce** execution rules, maintain **absolute consistency** in output quality, and ensure **full automation** without manual intervention.

=== PRIMARY OBJECTIVE ===
Generate **high-quality, strictly formatted Q&A** that is **ready for fine-tuning datasets**, while ensuring reliability, consistency, and minimal downtime.

=== QUALITY RULES ===
- Output must be **clear, precise, and professional**.
- No hallucinated data — answers must strictly match the input context.
- No filler text, disclaimers, or meta-commentary.
- Use **neutral, formal language**.
- Ensure proper Arabic diacritics are avoided unless explicitly required.
- Follow correct grammar, punctuation, and spelling.
- **NEVER** reduce quality or deviate from the defined Q&A structure.
- Reject and reprocess any output that does not meet the exact format.

=== GOAL ===
Produce **consistent Q&A datasets** ready for fine-tuning without any post-editing."""

    def _create_qa_prompt(self, text: str, language: str) -> str:
        """Create enhanced prompt for Q&A generation."""
        if language == "arabic":
            return f"""
قم بإنشاء أسئلة وأجوبة عالية الجودة ومتخصصة باللغة العربية من النص التالي.

المتطلبات:
- أنشئ بالضبط 5 أزواج من الأسئلة والأجوبة
- يجب أن تكون الأسئلة متنوعة ومتدرجة في الصعوبة
- تغطي المفاهيم الرئيسية والتفاصيل المهمة في النص
- استخدم لغة عربية فصحى واضحة ودقيقة
- تجنب التكرار في المعلومات بين الأسئلة
- اجعل الإجابات شاملة ودقيقة ومرتبطة مباشرة بالنص

النص:
{text}

قم بالرد بتنسيق JSON كما يلي بالضبط (بدون أي نص إضافي):
{{
  "qa_pairs": [
    {{
      "question": "السؤال هنا؟",
      "answer": "الإجابة هنا"
    }}
  ]
}}
"""
        else:
            return f"""
Generate high-quality, specialized question-answer pairs in English from the following text.

Requirements:
- Create exactly 5 question-answer pairs
- Questions should be diverse and varied in difficulty
- Cover key concepts and important details in the text
- Use clear, precise, and professional English
- Avoid repetition of information between questions
- Make answers comprehensive, accurate, and directly tied to the text

Text:
{text}

Respond in JSON format exactly as follows (no additional text):
{{
  "qa_pairs": [
    {{
      "question": "Question here?",
      "answer": "Answer here"
    }}
  ]
}}
"""
    
    def _get_current_model(self) -> str:
        """Get current model for API calls."""
        return self.available_models[self.current_model_index]

    def _switch_to_next_model(self) -> bool:
        """Switch to next available model. Returns True if switch successful."""
        if self.current_model_index < len(self.available_models) - 1:
            self.current_model_index += 1
            new_model = self._get_current_model()
            self.logger.warning(f"Switching to model: {new_model}")
            return True
        return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _call_gemini_api(self, prompt: str, language: str) -> Dict:
        """Make API call to Gemini with retry logic."""
        self._rate_limit()
        
        response = None
        current_model = self._get_current_model()
        
        try:
            enhanced_system_instruction = self._get_enhanced_system_instruction()
            
            self.logger.info(f"Using model: {current_model} for {language} Q&A generation")
            
            response = self.client.models.generate_content(
                model=current_model,
                contents=[
                    types.Content(role="user", parts=[types.Part(text=prompt)])
                ],
                config=types.GenerateContentConfig(
                    system_instruction=enhanced_system_instruction,
                    response_mime_type="application/json",
                    temperature=0.7,
                    max_output_tokens=3072  # Increased for longer responses
                )
            )
            
            if not response.text:
                self.logger.warning(f"Empty response from {current_model}, attempting model fallback")
                if self._switch_to_next_model():
                    raise ValueError(f"Empty response from {current_model}, retrying with next model")
                else:
                    raise ValueError("Empty response from Gemini API - all models exhausted")
            
            # Parse JSON response
            result = json.loads(response.text)
            
            if "qa_pairs" not in result:
                raise ValueError("Invalid response format: missing 'qa_pairs'")
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error with {current_model}: {e}")
            if response and response.text:
                self.logger.error(f"Raw response: {response.text}")
                # Try model fallback for JSON decode errors
                if self._switch_to_next_model():
                    raise ValueError(f"JSON decode error with {current_model}, retrying with next model")
            else:
                self.logger.error("No response received from API")
            raise
        except Exception as e:
            self.logger.error(f"API call failed with {current_model}: {e}")
            # Try model fallback for API errors
            if "timeout" in str(e).lower() or "rate" in str(e).lower() or "4" in str(e) or "5" in str(e):
                if self._switch_to_next_model():
                    raise ValueError(f"API error with {current_model}, retrying with next model") 
            raise
    
    def generate_qa_for_chunk(self, chunk: Dict, language: str) -> List[Dict]:
        """Generate Q&A pairs for a single chunk."""
        chunk_id = chunk['id']
        text = chunk['text']
        
        self.logger.info(f"Generating {language} Q&A for chunk {chunk_id}")
        
        try:
            prompt = self._create_qa_prompt(text, language)
            result = self._call_gemini_api(prompt, language)
            
            qa_pairs = result['qa_pairs']
            
            # Add metadata to each Q&A pair
            for qa in qa_pairs:
                qa['chunk_id'] = chunk_id
                qa['language'] = language
                qa['word_count'] = chunk['word_count']
            
            self.logger.info(f"Generated {len(qa_pairs)} Q&A pairs for {language} chunk {chunk_id}")
            return qa_pairs
            
        except Exception as e:
            self.logger.error(f"Failed to generate Q&A for {language} chunk {chunk_id}: {e}")
            return []
    
    def generate_qa_for_chunks(self, chunks: List[Dict], language: str) -> List[Dict]:
        """Generate Q&A pairs for all chunks in a language."""
        self.logger.info(f"Starting Q&A generation for {len(chunks)} {language} chunks")
        
        all_qa_pairs = []
        
        for i, chunk in enumerate(chunks):
            try:
                qa_pairs = self.generate_qa_for_chunk(chunk, language)
                all_qa_pairs.extend(qa_pairs)
                
                # Log progress
                progress = ((i + 1) / len(chunks)) * 100
                self.logger.info(f"Progress: {progress:.1f}% ({i + 1}/{len(chunks)} chunks)")
                
            except Exception as e:
                self.logger.error(f"Skipping chunk {chunk['id']} due to error: {e}")
                continue
        
        self.logger.info(f"Generated {len(all_qa_pairs)} total Q&A pairs for {language}")
        
        # Save intermediate results
        save_intermediate_file(all_qa_pairs, f"{language}_qa_pairs.json", "json")
        
        return all_qa_pairs
    
    def generate_bilingual_qa(self, aligned_chunks: List[tuple]) -> tuple:
        """Generate Q&A pairs for both languages from aligned chunks."""
        self.logger.info("Starting bilingual Q&A generation")
        
        # Extract Arabic and English chunks
        ar_chunks = [chunk_pair[0] for chunk_pair in aligned_chunks]
        en_chunks = [chunk_pair[1] for chunk_pair in aligned_chunks]
        
        # Generate Q&A for both languages
        ar_qa_pairs = self.generate_qa_for_chunks(ar_chunks, "arabic")
        en_qa_pairs = self.generate_qa_for_chunks(en_chunks, "english")
        
        self.logger.info("Bilingual Q&A generation completed")
        return ar_qa_pairs, en_qa_pairs
