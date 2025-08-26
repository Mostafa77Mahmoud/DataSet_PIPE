
#!/usr/bin/env python3
"""
Expert AI engineering dataset generation agent for AAOIFI judgmental verification dataset.
Generates high-quality examples with strict reference validation and balanced distribution.
"""

import json
import os
import time
import logging
import hashlib
import shutil
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
import random

try:
    import google.generativeai as genai
    import tiktoken
    from rapidfuzz import fuzz
    from tenacity import retry, stop_after_attempt, wait_exponential
    DEPENDENCIES_OK = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    DEPENDENCIES_OK = False

class DatasetGenerator:
    """High-quality judgmental dataset generator with strict validation."""
    
    def __init__(self, target_size: int = 3000, concurrency: int = 2):
        self.target_size = target_size
        self.concurrency = concurrency
        self.safety_margin = 0.96
        self.max_tokens_per_request = 250000
        
        # Token management
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Model configuration
        self.models = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
        self.current_model_idx = 0
        
        # Rate limiting
        self.requests_per_minute = 15
        self.last_request_time = 0
        
        # Setup logging
        self.setup_logging()
        
        # Load AAOIFI content
        self.aaoifi_content = self.load_aaoifi_content()
        
        # Stats tracking
        self.stats = {
            "total_generated": 0,
            "arabic_count": 0,
            "english_count": 0,
            "true_verdicts": 0,
            "false_verdicts": 0,
            "unknown_references": 0,
            "suspected_fabrications": 0
        }
    
    def setup_logging(self):
        """Setup comprehensive logging."""
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"logs/dataset_generation_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
    
    def create_backup(self) -> str:
        """Create timestamped backup of current data."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = f"backup/{timestamp}"
        
        os.makedirs(backup_dir, exist_ok=True)
        
        # Backup data and raw directories
        for dir_name in ["data", "raw"]:
            if os.path.exists(dir_name):
                shutil.copytree(dir_name, f"{backup_dir}/{dir_name}")
        
        self.logger.info(f"Created backup: {backup_dir}")
        return backup_dir
    
    def load_aaoifi_content(self) -> str:
        """Load canonical AAOIFI content."""
        content_file = "intermediate/aaofi_cleaned.txt"
        if not os.path.exists(content_file):
            self.logger.error(f"AAOIFI content not found: {content_file}")
            return ""
        
        with open(content_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.logger.info(f"Loaded AAOIFI content: {len(content)} characters")
        return content
    
    def check_environment(self) -> bool:
        """Check environment and dependencies."""
        if not DEPENDENCIES_OK:
            self.logger.error("Required dependencies not available")
            return False
        
        # Multiple API keys for high availability
        api_keys = [
            "AIzaSyBbidR_bEfiMrhOufE4PAHrYEBvuPuqakg",
            "AIzaSyAIPk1An1O6sZiro64Q4R9PjVrqvPkSVvQ"
        ]
        
        # Also check environment variable if available
        env_key = os.getenv('GEMINI_API_KEY')
        if env_key and env_key not in api_keys:
            api_keys.append(env_key)
        
        if not api_keys:
            self.logger.error("No API keys available")
            return False
        
        # Initialize API key manager
        from api_key_manager import APIKeyManager
        self.api_manager = APIKeyManager(api_keys)
        
        self.logger.info(f"Environment check passed with {len(api_keys)} API keys")
        return True
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def create_seeds(self) -> List[Dict]:
        """Create seed claims from AAOIFI content."""
        if not self.aaoifi_content:
            self.logger.error("No AAOIFI content available")
            return []
        
        seeds = []
        
        # Split content into paragraphs
        paragraphs = [p.strip() for p in self.aaoifi_content.split('\n\n') if p.strip()]
        
        seed_id = 0
        for chunk_id, paragraph in enumerate(paragraphs):
            if len(paragraph) < 50:  # Skip very short paragraphs
                continue
            
            # Extract potential claims from paragraph
            sentences = [s.strip() for s in re.split(r'[.!?]+', paragraph) if len(s.strip()) > 30]
            
            for sentence in sentences[:2]:  # Max 2 claims per paragraph
                # Detect language
                arabic_chars = len(re.findall(r'[\u0600-\u06FF]', sentence))
                is_arabic = arabic_chars > len(sentence) * 0.3
                
                language = "ar" if is_arabic else "en"
                
                seed = {
                    "seed_id": f"{language}_{seed_id:04d}",
                    "context": paragraph[:1000],  # Limit context size
                    "claim": sentence,
                    "chunk_id": chunk_id,
                    "language": language
                }
                seeds.append(seed)
                seed_id += 1
        
        # Balance languages in seeds
        arabic_seeds = [s for s in seeds if s['language'] == 'ar']
        english_seeds = [s for s in seeds if s['language'] == 'en']
        
        # Ensure minimum representation
        min_per_language = min(len(arabic_seeds), len(english_seeds), 500)
        balanced_seeds = arabic_seeds[:min_per_language] + english_seeds[:min_per_language]
        
        random.shuffle(balanced_seeds)
        
        # Save seeds
        os.makedirs("data", exist_ok=True)
        with open("data/judgmental_seed.jsonl", 'w', encoding='utf-8') as f:
            for seed in balanced_seeds:
                f.write(json.dumps(seed, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Created {len(balanced_seeds)} seeds ({len(arabic_seeds[:min_per_language])} Arabic, {len(english_seeds[:min_per_language])} English)")
        return balanced_seeds
    
    def get_verification_prompt(self, language: str) -> str:
        """Get verification prompt template."""
        if language == "ar":
            return """You are an expert AAOIFI verification auditor. Given a CONTEXT paragraph (from AAOIFI) and a short CLAIM (assertion derived from that text), produce a JSON object exactly matching the schema below. ONLY return valid JSON — no extra commentary.

Schema:
{{
"verdict": "True" | "False",
"explanation": "<one- or two-sentence justification that quotes or paraphrases the CONTEXT directly>",
"reference": "<exact citation found in the CONTEXT e.g. 'Standard 5, Clause 2/2'> or 'UNKNOWN'",
"language": "ar"
}}

RULES:
* Do NOT invent references. If the CONTEXT does not explicitly support/refute the claim, put "reference": "UNKNOWN" and set verdict according to available evidence.
* Use temperature 0.0.
* Keep explanation concise, and include a short quoted phrase from CONTEXT if applicable.

Input:
CONTEXT: {context}
CLAIM: {claim}"""
        else:
            return """You are an AAOIFI verification auditor. Given CONTEXT (from AAOIFI) and a CLAIM, return a single JSON object:

{{
"verdict": "True" | "False",
"explanation": "<concise justification, quoting the CONTEXT when possible>",
"reference": "<exact AAOIFI citation or 'UNKNOWN'>",
"language": "en"
}}

Rules:
* Do not fabricate references. If no citation can be found in CONTEXT, use "UNKNOWN".
* Use temperature 0.0, be concise, output only JSON.

Input:
CONTEXT: {context}
CLAIM: {claim}"""
    
    
    
    def call_gemini(self, prompt: str) -> Dict:
        """Make API call to Gemini with high-availability key rotation."""
        model_name = self.models[self.current_model_idx]
        
        try:
            # Use API manager for the call
            response = self.api_manager.make_api_call(
                model_name=model_name,
                prompt=prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=1000
                )
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"API call failed with {model_name}: {e}")
            
            # Try fallback model if quota/rate errors persist
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                if self.current_model_idx < len(self.models) - 1:
                    self.current_model_idx += 1
                    self.logger.warning(f"Switching to fallback model: {self.models[self.current_model_idx]}")
                    # Retry with new model
                    return self.call_gemini(prompt)
            
            raise
    
    def verify_reference(self, reference: str, context: str) -> Tuple[bool, bool]:
        """Verify reference against AAOIFI content."""
        if reference == "UNKNOWN":
            return True, False  # Valid but unknown
        
        # Check exact match (normalized)
        ref_normalized = re.sub(r'\s+', ' ', reference.lower().strip())
        content_normalized = re.sub(r'\s+', ' ', self.aaoifi_content.lower())
        
        if ref_normalized in content_normalized:
            return True, False
        
        # Fuzzy match with high threshold
        sentences = self.aaoifi_content.split('.')
        for sentence in sentences:
            if fuzz.ratio(ref_normalized, sentence.lower().strip()) >= 85:
                return True, False
        
        # Not found - suspected fabrication
        return False, True
    
    def process_batch(self, seeds: List[Dict], batch_id: int) -> List[Dict]:
        """Process a batch of seeds."""
        os.makedirs("raw", exist_ok=True)
        
        results = []
        
        for i, seed in enumerate(seeds):
            try:
                prompt = self.get_verification_prompt(seed['language']).format(
                    context=seed['context'],
                    claim=seed['claim']
                )
                
                # Make API call
                response = self.call_gemini(prompt)
                
                # Save raw response
                raw_file = f"raw/gemini_{response['model']}_batch_{batch_id}_seed_{seed['seed_id']}.json"
                with open(raw_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "seed": seed,
                        "prompt": prompt,
                        "response": response,
                        "timestamp": time.time()
                    }, f, ensure_ascii=False, indent=2)
                
                # Parse response
                response_text = response['text'].strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3]
                elif response_text.startswith('```'):
                    response_text = response_text[3:-3]
                
                data = json.loads(response_text)
                
                # Verify reference
                reference = data.get('reference', 'UNKNOWN')
                is_valid_ref, is_suspected_fabrication = self.verify_reference(reference, seed['context'])
                
                if not is_valid_ref and reference != 'UNKNOWN':
                    reference = 'UNKNOWN'
                    is_suspected_fabrication = True
                
                # Create final format
                result = {
                    "instruction": "Verify the following analysis against AAOIFI standards.",
                    "input": f"Claim: {seed['claim']}\nContext: {seed['context']}",
                    "output": f"VERDICT: {data['verdict']}\nExplanation: {data['explanation']}\nReference: {reference}",
                    "meta": {
                        "language": seed['language'],
                        "seed_id": seed['seed_id'],
                        "generator_model": response['model'],
                        "raw_response_path": raw_file,
                        "suspected_fabrication": is_suspected_fabrication,
                        "chunk_id": seed['chunk_id']
                    }
                }
                
                results.append(result)
                
                # Update stats
                self.stats['total_generated'] += 1
                if seed['language'] == 'ar':
                    self.stats['arabic_count'] += 1
                else:
                    self.stats['english_count'] += 1
                
                if data['verdict'] == 'True':
                    self.stats['true_verdicts'] += 1
                else:
                    self.stats['false_verdicts'] += 1
                
                if reference == 'UNKNOWN':
                    self.stats['unknown_references'] += 1
                
                if is_suspected_fabrication:
                    self.stats['suspected_fabrications'] += 1
                
                self.logger.info(f"Processed seed {seed['seed_id']}: {data['verdict']} ({seed['language']})")
                
            except Exception as e:
                self.logger.error(f"Failed to process seed {seed['seed_id']}: {e}")
                continue
        
        return results
    
    def calculate_batch_size(self, seeds: List[Dict]) -> int:
        """Calculate optimal batch size based on token limits."""
        if not seeds:
            return 0
        
        # Estimate tokens for a sample
        sample_prompt = self.get_verification_prompt(seeds[0]['language']).format(
            context=seeds[0]['context'],
            claim=seeds[0]['claim']
        )
        
        prompt_tokens = self.count_tokens(sample_prompt)
        effective_limit = int(self.max_tokens_per_request * self.safety_margin)
        
        # Reserve space for response
        available_tokens = effective_limit - 1000
        
        return max(1, min(len(seeds), available_tokens // prompt_tokens))
    
    def validate_progress(self) -> Dict:
        """Validate current progress against targets."""
        total = self.stats['total_generated']
        if total == 0:
            return {"valid": False, "reason": "No examples generated"}
        
        arabic_pct = (self.stats['arabic_count'] / total) * 100
        true_pct = (self.stats['true_verdicts'] / total) * 100
        unknown_pct = (self.stats['unknown_references'] / total) * 100
        
        validation = {
            "valid": True,
            "total": total,
            "arabic_percentage": arabic_pct,
            "true_percentage": true_pct,
            "unknown_percentage": unknown_pct,
            "needs_rebalancing": False,
            "targets_met": {
                "size": total >= self.target_size,
                "arabic_balance": arabic_pct >= 40.0,
                "verdict_balance": 47.0 <= true_pct <= 53.0,
                "reference_quality": unknown_pct <= 10.0
            }
        }
        
        if not all(validation["targets_met"].values()):
            validation["needs_rebalancing"] = True
        
        return validation
    
    def generate_dataset(self) -> bool:
        """Main generation workflow."""
        self.logger.info("🔍 STARTING DATASET GENERATION")
        self.logger.info(f"Target size: {self.target_size}")
        
        # 1. Environment & Safety
        if not self.check_environment():
            return False
        
        backup_dir = self.create_backup()
        
        # 2. Load & Seed Creation
        seeds = self.create_seeds()
        if not seeds:
            self.logger.error("No seeds created")
            return False
        
        # 3. Batch Generation
        all_results = []
        batch_id = 0
        processed_seeds = 0
        
        while processed_seeds < len(seeds) and self.stats['total_generated'] < self.target_size:
            remaining_seeds = seeds[processed_seeds:]
            batch_size = min(self.calculate_batch_size(remaining_seeds), 50)  # Max 50 per batch
            
            if batch_size == 0:
                break
            
            batch = remaining_seeds[:batch_size]
            
            self.logger.info(f"Processing batch {batch_id}: {len(batch)} seeds")
            batch_results = self.process_batch(batch, batch_id)
            all_results.extend(batch_results)
            
            processed_seeds += batch_size
            batch_id += 1
            
            # Validate progress every 100 examples
            if self.stats['total_generated'] % 100 == 0:
                validation = self.validate_progress()
                self.logger.info(f"Progress: {validation['total']}/{self.target_size} "
                               f"({validation['arabic_percentage']:.1f}% AR, "
                               f"{validation['true_percentage']:.1f}% True)")
            
            # Stop if target reached
            if self.stats['total_generated'] >= self.target_size:
                break
        
        # 4. Dedupe & Finalize
        self.deduplicate_and_finalize(all_results)
        
        # 5. Create splits
        self.create_train_val_test_splits()
        
        # 6. Human review sample
        self.create_human_review_sample()
        
        # 7. Final validation and report
        final_validation = self.validate_progress()
        self.create_final_report(final_validation)
        
        # 8. Report API usage statistics
        if hasattr(self, 'api_manager'):
            usage_stats = self.api_manager.get_usage_stats()
            self.logger.info(f"API Usage Summary: {usage_stats}")
        
        return final_validation['valid']
    
    def deduplicate_and_finalize(self, results: List[Dict]):
        """Deduplicate and create final dataset."""
        self.logger.info("Deduplicating and finalizing dataset...")
        
        # Hash-based deduplication
        seen_hashes = set()
        unique_results = []
        
        for result in results:
            content = result['instruction'] + result['input'] + result['output']
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_results.append(result)
        
        self.logger.info(f"Removed {len(results) - len(unique_results)} duplicates")
        
        # Shuffle and limit to target size
        random.shuffle(unique_results)
        final_results = unique_results[:self.target_size]
        
        # Save candidate file first
        candidate_file = "data/judgmental_final_candidate.jsonl"
        with open(candidate_file, 'w', encoding='utf-8') as f:
            for result in final_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # Validate candidate is not empty
        if len(final_results) > 0:
            os.rename(candidate_file, "data/judgmental_final.jsonl")
            self.logger.info(f"Saved final dataset: {len(final_results)} examples")
        else:
            self.logger.error("Final dataset is empty!")
    
    def create_train_val_test_splits(self):
        """Create stratified train/val/test splits."""
        if not os.path.exists("data/judgmental_final.jsonl"):
            return
        
        # Load final dataset
        examples = []
        with open("data/judgmental_final.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        
        # Stratify by language and verdict
        strata = defaultdict(list)
        for example in examples:
            language = example['meta']['language']
            verdict = 'True' if 'VERDICT: True' in example['output'] else 'False'
            strata[f"{language}_{verdict}"].append(example)
        
        # Create splits
        train_examples = []
        val_examples = []
        test_examples = []
        
        for stratum_examples in strata.values():
            random.shuffle(stratum_examples)
            n = len(stratum_examples)
            
            train_end = int(n * 0.8)
            val_end = int(n * 0.9)
            
            train_examples.extend(stratum_examples[:train_end])
            val_examples.extend(stratum_examples[train_end:val_end])
            test_examples.extend(stratum_examples[val_end:])
        
        # Shuffle final splits
        for split_examples in [train_examples, val_examples, test_examples]:
            random.shuffle(split_examples)
        
        # Save splits
        splits = {
            "train.jsonl": train_examples,
            "val.jsonl": val_examples,
            "test.jsonl": test_examples
        }
        
        for filename, examples in splits.items():
            with open(f"data/{filename}", 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Created splits: train={len(train_examples)}, "
                        f"val={len(val_examples)}, test={len(test_examples)}")
    
    def create_human_review_sample(self):
        """Create human review sample."""
        if not os.path.exists("data/judgmental_final.jsonl"):
            return
        
        # Load examples
        examples = []
        with open("data/judgmental_final.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        
        if not examples:
            return
        
        # Prioritize uncertain examples
        uncertain_examples = [e for e in examples if e['meta'].get('suspected_fabrication', False)]
        regular_examples = [e for e in examples if not e['meta'].get('suspected_fabrication', False)]
        
        # Sample 5% (min 20, max 500)
        sample_size = max(20, min(500, len(examples) // 20))
        uncertain_sample_size = min(len(uncertain_examples), sample_size // 2)
        regular_sample_size = sample_size - uncertain_sample_size
        
        sample = []
        if uncertain_examples:
            sample.extend(random.sample(uncertain_examples, uncertain_sample_size))
        if regular_examples and regular_sample_size > 0:
            sample.extend(random.sample(regular_examples, min(regular_sample_size, len(regular_examples))))
        
        # Save sample
        os.makedirs("review", exist_ok=True)
        with open("review/human_sample.jsonl", 'w', encoding='utf-8') as f:
            for example in sample:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Created human review sample: {len(sample)} examples")
    
    def create_final_report(self, validation: Dict):
        """Create comprehensive final report."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""# AAOIFI Judgmental Dataset Generation Report

Generated: {timestamp}
Target Size: {self.target_size}
Actual Size: {validation['total']}

## Distribution Analysis
- Arabic Examples: {self.stats['arabic_count']} ({validation['arabic_percentage']:.1f}%)
- English Examples: {self.stats['english_count']} ({100 - validation['arabic_percentage']:.1f}%)
- True Verdicts: {self.stats['true_verdicts']} ({validation['true_percentage']:.1f}%)
- False Verdicts: {self.stats['false_verdicts']} ({100 - validation['true_percentage']:.1f}%)

## Quality Metrics
- Unknown References: {self.stats['unknown_references']} ({validation['unknown_percentage']:.1f}%)
- Suspected Fabrications: {self.stats['suspected_fabrications']}

## Target Achievement
- Size Target: {'✅' if validation['targets_met']['size'] else '❌'} ({validation['total']}/{self.target_size})
- Arabic Balance: {'✅' if validation['targets_met']['arabic_balance'] else '❌'} ({validation['arabic_percentage']:.1f}% >= 40%)
- Verdict Balance: {'✅' if validation['targets_met']['verdict_balance'] else '❌'} ({validation['true_percentage']:.1f}% in 47-53% range)
- Reference Quality: {'✅' if validation['targets_met']['reference_quality'] else '❌'} ({validation['unknown_percentage']:.1f}% <= 10%)

## Files Generated
- data/judgmental_final.jsonl: Main dataset
- data/train.jsonl: Training split (80%)
- data/val.jsonl: Validation split (10%)  
- data/test.jsonl: Test split (10%)
- review/human_sample.jsonl: Human review sample

## Generation Log
{self.log_file}

Dataset generation {'completed successfully' if all(validation['targets_met'].values()) else 'completed with issues'}.
"""
        
        with open("SOLUTION_SUMMARY.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        with open("logs/generation_summary.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        self.logger.info("Generation completed - see SOLUTION_SUMMARY.md for full report")

def main():
    """Main execution function."""
    generator = DatasetGenerator(target_size=3000, concurrency=2)
    success = generator.generate_dataset()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
