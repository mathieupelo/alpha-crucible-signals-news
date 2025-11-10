#!/usr/bin/env python3
"""
Run all tests for alpha-crucible-signals-news.
"""

import sys
import subprocess

def run_test(test_file):
    """Run a test file and return success status."""
    print(f"\n{'='*60}")
    print(f"Running {test_file}")
    print('='*60)
    result = subprocess.run([sys.executable, test_file], capture_output=False)
    return result.returncode == 0

def main():
    """Run all tests."""
    tests = [
        'test_decay.py',
        'test_integration.py',
    ]
    
    results = []
    for test in tests:
        success = run_test(test)
        results.append((test, success))
    
    print(f"\n{'='*60}")
    print("Test Summary")
    print('='*60)
    
    all_passed = True
    for test, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test}")
        if not passed:
            all_passed = False
    
    print('='*60)
    if all_passed:
        print("All tests passed! ✓")
        return 0
    else:
        print("Some tests failed! ✗")
        return 1

if __name__ == "__main__":
    sys.exit(main())

