#!/usr/bin/env python3
"""
Fix Unicode encoding issues in test files for Windows compatibility.
"""

import re
import os

def fix_unicode_in_file(filepath):
    """Fix Unicode characters in a file."""
    print(f"Fixing Unicode in: {filepath}")
    
    # Unicode replacements
    unicode_replacements = {
        '🔧': '[TOOL]',
        '🧪': '[TEST]',
        '✅': '[PASS]',
        '❌': '[FAIL]',
        '📊': '[DATA]',
        '🎉': '[SUCCESS]',
        '⚠️': '[WARNING]',
        '🔍': '[SEARCH]',
        '🎯': '[TARGET]',
        '🚀': '[ROCKET]',
        '📋': '[LIST]',
        '💾': '[SAVE]',
        '🔴': '[CRITICAL]',
        '📈': '[CHART]',
        '🏗️': '[BUILD]',
        '🐛': '[BUG]',
        '🔧': '[TOOL]',
        '📝': '[NOTE]',
        '🎯': '[TARGET]',
        '🚀': '[ROCKET]',
        '📊': '[DATA]',
        '🔍': '[SEARCH]',
        '🧪': '[TEST]',
        '✅': '[PASS]',
        '❌': '[FAIL]',
        '⚠️': '[WARNING]',
        '🎉': '[SUCCESS]',
        '📋': '[LIST]',
        '💾': '[SAVE]',
        '🔴': '[CRITICAL]',
        '📈': '[CHART]',
        '🏗️': '[BUILD]',
        '🐛': '[BUG]',
        '📝': '[NOTE]',
        '🔧': '[TOOL]',
        '🎯': '[TARGET]',
        '🚀': '[ROCKET]',
        '📊': '[DATA]',
        '🔍': '[SEARCH]',
        '🧪': '[TEST]',
        '✅': '[PASS]',
        '❌': '[FAIL]',
        '⚠️': '[WARNING]',
        '🎉': '[SUCCESS]',
        '📋': '[LIST]',
        '💾': '[SAVE]',
        '🔴': '[CRITICAL]',
        '📈': '[CHART]',
        '🏗️': '[BUILD]',
        '🐛': '[BUG]',
        '📝': '[NOTE]',
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace Unicode characters
        for unicode_char, replacement in unicode_replacements.items():
            content = content.replace(unicode_char, replacement)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  Fixed Unicode characters in {filepath}")
        return True
        
    except Exception as e:
        print(f"  Error fixing {filepath}: {e}")
        return False

def main():
    """Fix Unicode in all test files."""
    test_files = [
        'test_inference_fix.py',
        'test_model_consistency.py',
        'test_training_consistency.py',
        'run_all_tests.py'
    ]
    
    print("Fixing Unicode encoding issues in test files...")
    
    for test_file in test_files:
        if os.path.exists(test_file):
            fix_unicode_in_file(test_file)
        else:
            print(f"File not found: {test_file}")
    
    print("Unicode fixing complete!")

if __name__ == "__main__":
    main()
