#!/usr/bin/env python3
"""
Pre-commit hook for Nasdaq Trader
Automatically runs security checks before commits
"""

import subprocess
import sys
import os

def run_security_check():
    """Run the security check script"""
    try:
        result = subprocess.run([sys.executable, 'security_check.py'], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print("SECURITY CHECK FAILED:")
            print(result.stdout)
            print(result.stderr)
            return False
        else:
            print("Security check passed")
            return True
    except Exception as e:
        print(f"Error running security check: {e}")
        return False

def main():
    """Main pre-commit hook function"""
    print("Running pre-commit security check...")
    
    if not run_security_check():
        print("\nCOMMIT BLOCKED: Security issues detected")
        print("Please fix security issues before committing")
        return 1
    
    print("Pre-commit checks passed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
