#!/usr/bin/env python3
"""
Security Check Script for Nasdaq Trader
Prevents committing sensitive data to repository
"""

import os
import re
import sys
from pathlib import Path

# Patterns that indicate potential security issues
SECURITY_PATTERNS = [
    r'api[_-]?key\s*=\s*["\']?[A-Za-z0-9_-]{20,}["\']?',
    r'secret\s*=\s*["\']?[A-Za-z0-9_-]{20,}["\']?',
    r'password\s*=\s*["\']?[A-Za-z0-9_-]{8,}["\']?',
    r'token\s*=\s*["\']?[A-Za-z0-9_-]{20,}["\']?',
    r'AIza[0-9A-Za-z_-]{35}',  # Google API key pattern
    r'sk-[0-9A-Za-z]{48}',     # OpenAI API key pattern
    r'xoxb-[0-9A-Za-z-]{10,}', # Slack bot token pattern
]

# Files to exclude from scanning
EXCLUDE_PATTERNS = [
    'security_check.py',
    '.git/',
    '__pycache__/',
    '.env.example',
    'env_example.txt',
    'SECURITY.md',
    'README.md',
]

def scan_file(file_path):
    """Scan a single file for security issues"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        issues = []
        for pattern in SECURITY_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                for match in matches:
                    # Check if it's not a placeholder
                    if not any(placeholder in match.lower() for placeholder in 
                             ['your_', 'placeholder', 'example', 'template', 'replace']):
                        issues.append({
                            'pattern': pattern,
                            'match': match,
                            'line': content[:content.find(match)].count('\n') + 1
                        })
        
        return issues
    except Exception as e:
        print(f"Error scanning {file_path}: {e}")
        return []

def should_exclude_file(file_path):
    """Check if file should be excluded from scanning"""
    file_str = str(file_path)
    return any(pattern in file_str for pattern in EXCLUDE_PATTERNS)

def main():
    """Main security check function"""
    print("Running security check...")
    
    issues_found = []
    
    # Scan all files in the repository
    for root, dirs, files in os.walk('.'):
        # Skip .git directory
        if '.git' in dirs:
            dirs.remove('.git')
            
        for file in files:
            file_path = Path(root) / file
            
            if should_exclude_file(file_path):
                continue
                
            if file_path.suffix in ['.py', '.txt', '.json', '.yaml', '.yml', '.md']:
                issues = scan_file(file_path)
                if issues:
                    issues_found.extend([(file_path, issue) for issue in issues])
    
    if issues_found:
        print("SECURITY ISSUES DETECTED:")
        print("=" * 50)
        
        for file_path, issue in issues_found:
            print(f"File: {file_path}")
            print(f"Line {issue['line']}: {issue['match']}")
            print(f"Pattern: {issue['pattern']}")
            print("-" * 30)
        
        print("\nCOMMIT BLOCKED: Remove sensitive data before committing")
        return 1
    else:
        print("Security check passed - no sensitive data detected")
        return 0

if __name__ == "__main__":
    sys.exit(main())
