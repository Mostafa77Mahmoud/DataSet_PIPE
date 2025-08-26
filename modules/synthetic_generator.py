
import json
import logging
import time
import os
import hashlib
from typing import List, Dict, Optional, Tuple, Any
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken
from modules.utils import save_intermediate_file

class TokenConfig:
    """Configuration for token management."""
    def __init__(self, 
                 max_tokens_per_request: int = 250000,
                 safety_margin: float = 0.96,
                 per_chunk_expected_output: int = 512):
        self.max_tokens_per_request = max_tokens_per_request
        self.safety_margin = safety_margin
        self.per_chunk_expected_output = per_chunk_expected_output
        self.effective_limit = int(max_tokens_per_request * safety_margin)

class SyntheticGenerator:
    """Generate synthetic judgmental examples using Gemini API."""
    
    def __init__(self, 
                 api_key: str = None,
                 api_keys: List[str] = None, 
                 models: List[str] = None,
                 token_config: TokenConfig = None,
                 paraphrase_temp: float = 0.2):
        
        # Handle multiple API keys for high availability
        if api_keys:
            from api_key_manager import APIKeyManager
            self.api_manager = APIKeyManager(api_keys)
        elif api_key:
            from api_key_manager import APIKeyManager
            self.api_manager = APIKeyManager([api_key])
        else:
            # Use default keys
            default_keys = [
                "AIzaSyBbidR_bEfiMrhOufE4PAHrYEBvuPuqakg",
                "AIzaSyAIPk1An1O6sZiro64Q4R9PjVrqvPkSVvQ"
            ]
            from api_key_manager import APIKeyManager
            self.api_manager = APIKeyManager(default_keys)
        
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.available_models = models or ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
        self.current_model_index = 0
        self.paraphrase_temp = paraphrase_temp
        
        # Token management
        self.token_config = token_config or TokenConfig()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Rate limiting - reduced for quota management
        self.requests_per_minute = 15
        self.last_request_time = 0
        
        # Generation tracking
        self.generated_count = 0
        self.target_size = 0
        self.correct_count = 0
        self.incorrect_count = 0
        
        # Create directories
        os.makedirs("raw", exist_ok=True)
        os.makedirs("intermediate", exist_ok=True)
        os.makedirs("data", exist_ok=True)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))
    
    def _rate_limit(self):
        """Implement rate limiting for API calls."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        min_interval = 60.0 / self.requests_per_minute
        
        if time_since_last_request < min_interval:
            sleep_time = min_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_current_model(self) -> str:
        """Get current model for API calls."""
        return self.available_models[self.current_model_index]
    
    def _switch_to_next_model(self) -> bool:
        """Switch to next available model."""
        if self.current_model_index < len(self.available_models) - 1:
            self.current_model_index += 1
            new_model = self._get_current_model()
            self.logger.warning(f"Switching to model: {new_model}")
            return True
        return False
    
    def _get_generation_prompt(self, language: str, seed_data: Dict) -> str:
        """Get generation prompt for given language."""
        
        if language == "en":
            return f"""You are a strict verifier for AAOIFI Shari'ah Standards. Given the following seed claim and optional context, produce a JSON array of exactly 3-5 verification examples. Each example must follow this schema exactly:
{{
  "type": "correct"|"incorrect",
  "claim": "<claim text>",
  "explanation": "<one-paragraph justification with citations>",
  "reference": "Standard No. X, clause Y/Z",
  "language": "en"
}}

CRITICAL RULES - NO EXCEPTIONS:
- 50% of examples across the dataset must be type "correct" and 50% "incorrect".
- If you cannot find or confirm an exact reference inside the provided AAOIFI context, you MUST label the claim as "incorrect" and MUST NOT invent a reference.
- If you cannot provide a true reference from the context, set reference to "UNKNOWN" and the example will be marked as suspected_fabrication.
- Only use references that exist verbatim or can be directly inferred from the provided context.
- Incorrect examples must cover varied error types: wrong standard number, wrong clause, incorrect interpretation (permits↔prohibits), overgeneralization, temporal/context mismatch.
- Use temperature=0.0 for correctness (0.2 for paraphrases only).
- Return strict JSON only. No surrounding commentary.

Seed Claim: {seed_data['claim']}
Context: {seed_data['context']}"""
        
        else:  # Arabic
            return f"""أنت مُدقّق مختص بمعايير أيوفي. بناءً على البيان (Claim) والنص السياقي، أعد مصفوفة JSON من 3-5 أمثلة تحقق. كل عنصر يجب أن يكون بالشكل:
{{
  "type": "correct"|"incorrect",
  "claim": "<النص الادعائي>",
  "explanation": "<تبرير مختصر مع استشهاد>",
  "reference": "المعيار رقم X، البند Y/Z",
  "language": "ar"
}}

قواعد أساسية - بدون استثناءات:
- 50% أمثلة صحيحة و50% خاطئة عبر مجموعة البيانات.
- إذا لم تتمكن من العثور على مرجع دقيق أو تأكيده داخل السياق المقدم لأيوفي، يجب أن تصنف الادعاء كـ "incorrect" ولا يجب أن تخترع مرجعاً.
- إذا لم تتمكن من تقديم مرجع حقيقي من السياق، اضبط المرجع على "UNKNOWN" وسيتم وضع علامة على المثال كـ suspected_fabrication.
- استخدم فقط المراجع الموجودة حرفياً أو التي يمكن استنتاجها مباشرة من السياق المقدم.
- الأخطاء يجب أن تشمل: رقم معيار خاطئ، بند خاطئ، تفسير خاطئ (جواز ↔ تحريم)، تعميم زائد، عدم توافق زمني/سياقي.
- أرجع JSON صالح فقط. استخدم temperature=0.0 (0.2 فقط للبارافريز).

البيان: {seed_data['claim']}
السياق: {seed_data['context']}"""
    
    def _calculate_batch_size(self, seeds: List[Dict]) -> int:
        """Calculate optimal batch size based on token limits."""
        if not seeds:
            return 0
        
        # Estimate tokens for a single seed
        sample_seed = seeds[0]
        prompt = self._get_generation_prompt(sample_seed['language'], sample_seed)
        prompt_tokens = self._count_tokens(prompt)
        
        # Reserve tokens for expected output
        available_tokens = self.token_config.effective_limit - self.token_config.per_chunk_expected_output
        
        # Calculate how many seeds can fit
        max_seeds = max(1, available_tokens // prompt_tokens)
        
        return min(len(seeds), max_seeds)
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=3, max=60)
    )
    def _call_gemini_api(self, prompt: str) -> Dict:
        """Make API call to Gemini with retry logic."""
        self._rate_limit()
        
        current_model = self._get_current_model()
        
        try:
            self.logger.info(f"Using model: {current_model} for synthetic generation")
            
            response = self.client.models.generate_content(
                model=current_model,
                contents=[
                    types.Content(role="user", parts=[types.Part(text=prompt)])
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,  # Strict for label generation
                    max_output_tokens=2048  # Increased to get more examples per request
                )
            )
            
            if not response.text:
                self.logger.warning(f"Empty response from {current_model}")
                if self._switch_to_next_model():
                    raise ValueError(f"Empty response from {current_model}, retrying with next model")
                else:
                    raise ValueError("Empty response from all models")
            
            # Parse JSON response
            result = json.loads(response.text)
            
            if not isinstance(result, list):
                raise ValueError("Response must be a JSON array")
            
            return {"examples": result, "model": current_model}
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error with {current_model}: {e}")
            if self._switch_to_next_model():
                raise ValueError(f"JSON decode error with {current_model}, retrying with next model")
            raise
        except Exception as e:
            self.logger.error(f"API call failed with {current_model}: {e}")
            if "timeout" in str(e).lower() or "rate" in str(e).lower():
                if self._switch_to_next_model():
                    raise ValueError(f"API error with {current_model}, retrying with next model")
            raise
    
    def _process_batch(self, seeds: List[Dict], batch_id: int) -> List[Dict]:
        """Process a batch of seeds."""
        if not seeds:
            return []
        
        self.logger.info(f"Processing batch {batch_id} with {len(seeds)} seeds")
        
        all_examples = []
        
        for seed in seeds:
            try:
                prompt = self._get_generation_prompt(seed['language'], seed)
                result = self._call_gemini_api(prompt)
                
                # Save raw response
                raw_file = f"raw/gemini_{result['model']}_batch_{batch_id}_seed_{seed['seed_id']}.json"
                with open(raw_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "seed_id": seed['seed_id'],
                        "prompt": prompt,
                        "response": result,
                        "timestamp": time.time()
                    }, f, ensure_ascii=False, indent=2)
                
                # Process examples
                examples = result['examples']
                processed_examples = []
                
                for example in examples:
                    # Convert to final format
                    final_example = self._convert_to_final_format(example, seed, result['model'], raw_file)
                    processed_examples.append(final_example)
                    
                    # Update counts
                    if example.get('type') == 'correct':
                        self.correct_count += 1
                    else:
                        self.incorrect_count += 1
                    
                    self.generated_count += 1
                
                all_examples.extend(processed_examples)
                
                # Save per-seed intermediate
                intermediate_file = f"intermediate/synthetic_by_seed_{seed['seed_id']}.json"
                with open(intermediate_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_examples, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"Generated {len(processed_examples)} examples for seed {seed['seed_id']}")
                
                # Check if we've reached target
                if self.target_size > 0 and self.generated_count >= self.target_size:
                    self.logger.info(f"Reached target size of {self.target_size}")
                    break
                    
            except Exception as e:
                self.logger.error(f"Failed to process seed {seed['seed_id']}: {e}")
                # Mark seed with failure
                seed['meta']['generator_failure'] = True
                continue
        
        return all_examples
    
    def _convert_to_final_format(self, example: Dict, seed: Dict, model: str, raw_file: str) -> Dict:
        """Convert generated example to final Alpaca format."""
        
        # Create prompt hash
        prompt_hash = hashlib.sha256(
            f"{seed['claim']}{seed['context']}".encode('utf-8')
        ).hexdigest()[:16]
        
        # Determine verdict
        verdict = "True" if example.get('type') == 'correct' else "False"
        
        # Check for suspected fabrication
        reference = example.get('reference', '')
        suspected_fabrication = reference == "UNKNOWN" or (verdict == "True" and not reference)
        
        # Format output
        output = f"VERDICT: {verdict}\n"
        output += f"Explanation: {example.get('explanation', '')}\n"
        output += f"Reference: {reference if reference != 'UNKNOWN' else 'Context'}"
        
        # Format input
        input_text = f"Claim: {example.get('claim', seed['claim'])}\n"
        input_text += f"Context: {seed['context']}"
        
        return {
            "instruction": "Verify the following analysis against AAOIFI standards.",
            "input": input_text,
            "output": output,
            "meta": {
                "language": seed['language'],
                "seed_id": seed['seed_id'],
                "generator": model,
                "prompt_hash": prompt_hash,
                "raw_response": raw_file,
                "suspected_fabrication": suspected_fabrication,
                "type": example.get('type', 'unknown'),
                "reference": reference,
                "original_seed": seed
            }
        }
    
    def generate_synthetic_dataset(self, 
                                 seed_file: str, 
                                 target_size: int = 10000,
                                 resume: bool = False) -> List[Dict]:
        """Generate synthetic dataset from seed file."""
        
        self.target_size = target_size
        self.logger.info(f"Starting synthetic generation with target size: {target_size}")
        
        # Load seed data
        seeds = []
        with open(seed_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    seeds.append(json.loads(line))
        
        self.logger.info(f"Loaded {len(seeds)} seed entries")
        
        # Process in batches
        all_examples = []
        batch_id = 0
        
        i = 0
        while i < len(seeds) and (self.target_size == 0 or self.generated_count < self.target_size):
            # Calculate batch size
            remaining_seeds = seeds[i:]
            batch_size = self._calculate_batch_size(remaining_seeds)
            
            if batch_size == 0:
                break
            
            # Get batch
            batch = remaining_seeds[:batch_size]
            
            # Process batch
            batch_examples = self._process_batch(batch, batch_id)
            all_examples.extend(batch_examples)
            
            # Update progress
            progress = (self.generated_count / self.target_size * 100) if self.target_size > 0 else 0
            self.logger.info(f"Progress: {progress:.1f}% ({self.generated_count}/{self.target_size if self.target_size > 0 else 'unlimited'})")
            
            i += batch_size
            batch_id += 1
            
            # Stop if target reached
            if self.target_size > 0 and self.generated_count >= self.target_size:
                break
        
        self.logger.info(f"Generated {len(all_examples)} total examples")
        self.logger.info(f"Correct: {self.correct_count}, Incorrect: {self.incorrect_count}")
        
        return all_examples
    
    def save_dataset(self, examples: List[Dict], output_file: str) -> None:
        """Save generated dataset to file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Saved {len(examples)} examples to {output_file}")
