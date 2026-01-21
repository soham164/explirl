#!/usr/bin/env python3
"""
PyPI Readiness Validation Script for expliRL
============================================
This script validates that expliRL is ready for PyPI publication.
"""

import sys
import os
import subprocess
import importlib.util

def check_file_exists(filepath, description):
    """Check if a required file exists"""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} - MISSING")
        return False

def check_file_content(filepath, required_content, description):
    """Check if file contains required content"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if required_content in content:
                print(f"✓ {description}")
                return True
            else:
                print(f"✗ {description} - MISSING CONTENT")
                return False
    except Exception as e:
        print(f"✗ {description} - ERROR: {e}")
        return False

def check_import():
    """Test that the package imports correctly"""
    try:
        import expliRL
        from expliRL import SHAPExplainer, LIMEExplainer, CounterfactualExplainer, RLCounterfactualExplainer
        print(f"✓ Package imports successfully (version: {expliRL.__version__})")
        return True
    except Exception as e:
        print(f"✗ Package import failed: {e}")
        return False

def check_dependencies():
    """Check that dependencies are properly resolved"""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'check'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ All dependencies resolved correctly")
            return True
        else:
            print(f"✗ Dependency issues: {result.stdout}")
            return False
    except Exception as e:
        print(f"✗ Error checking dependencies: {e}")
        return False

def check_cli():
    """Test CLI functionality"""
    try:
        result = subprocess.run(['explirl', '--help'], 
                              capture_output=True, text=True)
        if result.returncode == 0 and 'expliRL CLI' in result.stdout:
            print("✓ CLI tool works correctly")
            return True
        else:
            print(f"✗ CLI tool failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ CLI test error: {e}")
        return False

def main():
    """Run all PyPI readiness checks"""
    print("🔍 expliRL PyPI Readiness Validation")
    print("=" * 50)
    
    checks = []
    
    # File existence checks
    print("\n📁 Required Files:")
    checks.append(check_file_exists("setup.py", "Setup script"))
    checks.append(check_file_exists("README.md", "README file"))
    checks.append(check_file_exists("LICENSE", "License file"))
    checks.append(check_file_exists("requirements.txt", "Requirements file"))
    checks.append(check_file_exists("MANIFEST.in", "Manifest file"))
    
    # Content checks
    print("\n📝 Content Validation:")
    checks.append(check_file_content("setup.py", "expliRL Contributors", "Author info in setup.py"))
    checks.append(check_file_content("setup.py", "gymnasium>=0.26.0", "Gymnasium dependency"))
    checks.append(check_file_content("LICENSE", "MIT License", "MIT License content"))
    checks.append(check_file_content("setup.py", "github.com/explirl/expliRL", "GitHub URL"))
    
    # Functional checks
    print("\n🔧 Functionality Tests:")
    checks.append(check_import())
    checks.append(check_dependencies())
    checks.append(check_cli())
    
    # Summary
    print("\n" + "=" * 50)
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print(f"🎉 ALL CHECKS PASSED ({passed}/{total})")
        print("✅ expliRL is READY for PyPI publication!")
        print("\nNext steps:")
        print("1. python setup.py sdist bdist_wheel")
        print("2. twine check dist/*")
        print("3. twine upload --repository-url https://test.pypi.org/legacy/ dist/*")
        print("4. twine upload dist/*")
        return 0
    else:
        print(f"❌ CHECKS FAILED ({passed}/{total})")
        print("🔧 Please fix the issues above before publishing to PyPI")
        return 1

if __name__ == "__main__":
    sys.exit(main())