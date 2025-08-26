
#!/usr/bin/env python3
"""
Run Qwen analysis on judgmental dataset
"""

import json
import os
from collections import Counter

def load_jsonl(file_path):
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def mock_qwen_analysis(example):
    """Mock Qwen analysis - replace with actual Qwen API call."""
    # This is a placeholder - in real implementation, you would call Qwen API
    import random
    
    # Extract claim and context
    input_text = example.get('input', '')
    claim = ""
    if 'Claim:' in input_text:
        claim = input_text.split('Claim:')[1].split('\n')[0].strip()
    
    # Mock analysis based on simple heuristics
    confidence = random.uniform(0.7, 0.95)
    
    # Simple heuristic: if claim contains negation words, likely False
    negation_words = ['not', 'never', 'cannot', 'incorrect', 'wrong', 'false', 'لا', 'ليس', 'غير']
    has_negation = any(word in claim.lower() for word in negation_words)
    
    if has_negation:
        verdict = "False"
        confidence *= 0.9
    else:
        verdict = "True"
    
    return {
        'qwen_verdict': verdict,
        'qwen_confidence': confidence,
        'claim': claim
    }

def main():
    # Create analysis directory
    os.makedirs('analysis', exist_ok=True)
    
    # Load dataset
    data = load_jsonl('data/judgmental_final.jsonl')
    print(f"Analyzing {len(data)} examples with Qwen...")
    
    qwen_results = []
    disagreements = []
    
    for i, example in enumerate(data):
        # Get dataset verdict
        output = example.get('output', '')
        dataset_verdict = "True" if "VERDICT: True" in output else "False"
        
        # Run Qwen analysis
        qwen_result = mock_qwen_analysis(example)
        qwen_verdict = qwen_result['qwen_verdict']
        
        # Create result record
        result = {
            'seed_id': example.get('meta', {}).get('seed_id', ''),
            'language': example.get('meta', {}).get('language', ''),
            'dataset_verdict': dataset_verdict,
            'qwen_verdict': qwen_verdict,
            'qwen_confidence': qwen_result['qwen_confidence'],
            'claim': qwen_result['claim'],
            'agreement': dataset_verdict == qwen_verdict
        }
        
        qwen_results.append(result)
        
        # Track disagreements
        if not result['agreement']:
            disagreements.append(result)
        
        if i % 100 == 0:
            print(f"Progress: {i}/{len(data)}")
    
    # Save results
    with open('analysis/qwen_results.jsonl', 'w', encoding='utf-8') as f:
        for result in qwen_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    with open('analysis/qwen_disagreements.jsonl', 'w', encoding='utf-8') as f:
        for disagreement in disagreements:
            f.write(json.dumps(disagreement, ensure_ascii=False) + '\n')
    
    # Calculate summary stats
    total = len(qwen_results)
    agreements = sum(1 for r in qwen_results if r['agreement'])
    agreement_rate = agreements / total if total > 0 else 0
    
    summary = {
        'total_examples': total,
        'agreements': agreements,
        'disagreements': len(disagreements),
        'agreement_rate': agreement_rate,
        'verdict_distribution': {
            'dataset': Counter(r['dataset_verdict'] for r in qwen_results),
            'qwen': Counter(r['qwen_verdict'] for r in qwen_results)
        },
        'avg_confidence': sum(r['qwen_confidence'] for r in qwen_results) / total if total > 0 else 0
    }
    
    with open('analysis/qwen_agreement_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"Qwen analysis complete:")
    print(f"- Total examples: {total}")
    print(f"- Agreements: {agreements}")
    print(f"- Disagreements: {len(disagreements)}")
    print(f"- Agreement rate: {agreement_rate:.2%}")
    print(f"- Average confidence: {summary['avg_confidence']:.3f}")

if __name__ == "__main__":
    main()
