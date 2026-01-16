#!/usr/bin/env python3
"""
Quick validation script to verify the bot structure
This doesn't require external dependencies
"""

import sys
import os

def validate_structure():
    """Validate that all necessary files exist"""
    required_files = [
        'apex_bot/__init__.py',
        'apex_bot/config.py',
        'apex_bot/neural_network.py',
        'apex_bot/trading_agent.py',
        'apex_bot/data_fetcher.py',
        'apex_bot/utils.py',
        'main.py',
        'requirements.txt',
        'setup.py',
        'README.md',
        '.gitignore',
        '.env.example'
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
    
    if missing:
        print("❌ Missing files:")
        for f in missing:
            print(f"  - {f}")
        return False
    
    print("✅ All required files present")
    return True


def validate_imports():
    """Validate that Python files have valid syntax"""
    import py_compile
    
    python_files = [
        'apex_bot/__init__.py',
        'apex_bot/config.py',
        'apex_bot/neural_network.py',
        'apex_bot/trading_agent.py',
        'apex_bot/data_fetcher.py',
        'apex_bot/utils.py',
        'main.py',
        'setup.py'
    ]
    
    errors = []
    for file in python_files:
        try:
            py_compile.compile(file, doraise=True)
        except py_compile.PyCompileError as e:
            errors.append((file, str(e)))
    
    if errors:
        print("❌ Syntax errors found:")
        for file, error in errors:
            print(f"  {file}: {error}")
        return False
    
    print("✅ All Python files have valid syntax")
    return True


def main():
    print("="*60)
    print("APEX Trading Bot - Structure Validation")
    print("="*60)
    print()
    
    structure_ok = validate_structure()
    syntax_ok = validate_imports()
    
    print()
    if structure_ok and syntax_ok:
        print("✅ Validation passed! Bot structure is ready.")
        print()
        print("Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Copy .env.example to .env and configure")
        print("3. Run demo: python main.py --mode demo --symbol BTC/USD")
        return 0
    else:
        print("❌ Validation failed. Please fix the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
