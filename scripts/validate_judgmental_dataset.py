
#!/usr/bin/env python3
"""
Comprehensive validation script for judgmental datasets.
"""

import json
import os
import re
from collections import Counter
from typing import Dict, List, Any, Tuple
import logging

def setup_logging():
    """Setup logging for validation."""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def validate_entry_structure(entry: Dict[str, Any], line_num: int) -> List[str]:
    """Validate structure of a single entry."""
    issues = []
    
    # Required fields for Alpaca format
    required_fields = ['instruction', 'input', 'output']
    for field in required_fields:
        if field not in entry:
            issues.append(f"Line {line_num}: Missing required field '{field}'")
    
    if 'output' in entry:
        output = entry['output']
        
        # Check for VERDICT
        if not re.search(r'VERDICT:\s*(True|False|correct|incorrect)', output, re.IGNORECASE):
            issues.append(f"Line {line_num}: Missing or invalid VERDICT format")
        
        # Check for Explanation
        if not re.search(r'(Explanation|تفسير|توضيح):', output, re.IGNORECASE):
            issues.append(f"Line {line_num}: Missing explanation")
        
        # Check for Reference
        if not re.search(r'(Reference|مرجع):', output, re.IGNORECASE):
            issues.append(f"Line {line_num}: Missing reference")
        
        # Flag potential fabricated references
        if re.search(r'Reference:\s*(?!UNKNOWN).*', output) and 'UNKNOWN' not in output:
            # This should be manually checked
            pass
    
    return issues

def analyze_dataset_balance(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze dataset balance and distribution."""
    
    verdicts = Counter()
    languages = Counter()
    total_entries = len(entries)
    
    for entry in entries:
        output = entry.get('output', '').lower()
        
        # Count verdicts
        if 'verdict: true' in output or 'correct' in output:
            verdicts['true'] += 1
        else:
            verdicts['false'] += 1
        
        # Count languages
        instruction = entry.get('instruction', '')
        if any(arabic_char in instruction for arabic_char in 'ابتثجحخدذرزسشصضطظعغفقكلمنهوي'):
            languages['arabic'] += 1
        else:
            languages['english'] += 1
    
    balance_ratio = min(verdicts.values()) / max(verdicts.values()) if verdicts and max(verdicts.values()) > 0 else 0
    
    return {
        'total_entries': total_entries,
        'verdicts': dict(verdicts),
        'languages': dict(languages),
        'balance_ratio': balance_ratio,
        'verdict_percentages': {k: (v/total_entries)*100 for k, v in verdicts.items()},
        'language_percentages': {k: (v/total_entries)*100 for k, v in languages.items()}
    }

def validate_dataset(file_path: str) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    """Validate entire dataset file."""
    
    if not os.path.exists(file_path):
        return [], [f"File {file_path} does not exist"], {}
    
    entries = []
    all_issues = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                    
                    # Validate structure
                    issues = validate_entry_structure(entry, line_num)
                    all_issues.extend(issues)
                    
                except json.JSONDecodeError as e:
                    all_issues.append(f"Line {line_num}: Invalid JSON - {e}")
    
    # Analyze balance
    analysis = analyze_dataset_balance(entries)
    
    return entries, all_issues, analysis

def generate_validation_report(file_path: str, entries: List[Dict], issues: List[str], analysis: Dict) -> str:
    """Generate comprehensive validation report."""
    
    report = f"""
# Judgmental Dataset Validation Report

**Dataset:** {file_path}
**Generated:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics
- **Total Entries:** {analysis.get('total_entries', 0)}
- **Valid Entries:** {len(entries)}
- **Issues Found:** {len(issues)}

## Balance Analysis
### Verdict Distribution
- **True:** {analysis.get('verdicts', {}).get('true', 0)} ({analysis.get('verdict_percentages', {}).get('true', 0):.1f}%)
- **False:** {analysis.get('verdicts', {}).get('false', 0)} ({analysis.get('verdict_percentages', {}).get('false', 0):.1f}%)
- **Balance Ratio:** {analysis.get('balance_ratio', 0):.3f}

### Language Distribution  
- **Arabic:** {analysis.get('languages', {}).get('arabic', 0)} ({analysis.get('language_percentages', {}).get('arabic', 0):.1f}%)
- **English:** {analysis.get('languages', {}).get('english', 0)} ({analysis.get('language_percentages', {}).get('english', 0):.1f}%)

## Quality Assessment
"""

    if analysis.get('balance_ratio', 0) >= 0.8:
        report += "✅ **Balance:** Excellent (>= 80%)\n"
    elif analysis.get('balance_ratio', 0) >= 0.6:
        report += "⚠️ **Balance:** Good (>= 60%)\n"
    else:
        report += "❌ **Balance:** Poor (< 60%)\n"

    if len(issues) == 0:
        report += "✅ **Structure:** No issues found\n"
    elif len(issues) <= 5:
        report += f"⚠️ **Structure:** {len(issues)} minor issues\n"
    else:
        report += f"❌ **Structure:** {len(issues)} issues found\n"

    if issues:
        report += "\n## Issues Found\n"
        for issue in issues[:20]:  # Show first 20 issues
            report += f"- {issue}\n"
        if len(issues) > 20:
            report += f"- ... and {len(issues) - 20} more issues\n"

    report += f"""
## Recommendations
1. Target balance ratio should be >= 0.8 (currently {analysis.get('balance_ratio', 0):.3f})
2. All entries should have proper VERDICT, Explanation, and Reference format
3. References should be 'UNKNOWN' if uncertain (no fabrication)
4. Maintain bilingual distribution based on source content

## Dataset Status
"""
    
    if len(issues) == 0 and analysis.get('balance_ratio', 0) >= 0.8:
        report += "🎉 **READY FOR TRAINING**\n"
    elif len(issues) <= 5 and analysis.get('balance_ratio', 0) >= 0.6:
        report += "⚠️ **NEEDS MINOR FIXES**\n"
    else:
        report += "❌ **NEEDS SIGNIFICANT FIXES**\n"
    
    return report

def main():
    """Main validation function."""
    
    logger = setup_logging()
    
    # Files to validate
    datasets_to_validate = [
        "data/judgmental_final.jsonl",
        "data/train.jsonl", 
        "data/val.jsonl",
        "data/test.jsonl"
    ]
    
    os.makedirs("logs", exist_ok=True)
    
    for dataset_file in datasets_to_validate:
        if os.path.exists(dataset_file):
            logger.info(f"Validating {dataset_file}")
            
            entries, issues, analysis = validate_dataset(dataset_file)
            
            # Generate report
            report = generate_validation_report(dataset_file, entries, issues, analysis)
            
            # Save report
            report_file = f"logs/validation_report_{os.path.basename(dataset_file).replace('.jsonl', '')}.json"
            
            report_data = {
                "file": dataset_file,
                "timestamp": __import__('datetime').datetime.now().isoformat(),
                "entries_count": len(entries),
                "issues_count": len(issues),
                "issues": issues,
                "analysis": analysis,
                "report": report
            }
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Report saved to {report_file}")
            print(report)
            print("="*80)
        else:
            logger.warning(f"Dataset {dataset_file} not found")
    
    # Generate overall summary
    summary_path = "logs/validation_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Overall Validation Summary\n\n")
        f.write(f"Validation completed at: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for dataset_file in datasets_to_validate:
            if os.path.exists(dataset_file):
                f.write(f"- {dataset_file}: ✅ Validated\n")
            else:
                f.write(f"- {dataset_file}: ❌ Not found\n")
    
    logger.info(f"Overall summary saved to {summary_path}")

if __name__ == "__main__":
    main()
