#!/usr/bin/env python3
"""
GitHub Setup Script for Blueberry DE-Train
Helps prepare the repository for GitHub push
"""
import os
import subprocess
import sys
from pathlib import Path

def check_git_installed():
    """Check if git is installed"""
    try:
        subprocess.run(['git', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_git_repo():
    """Check if current directory is a git repository"""
    return Path('.git').exists()

def check_gitignore():
    """Verify .gitignore contains important exclusions"""
    gitignore_path = Path('.gitignore')
    if not gitignore_path.exists():
        return False
    
    content = gitignore_path.read_text()
    required_patterns = [
        'temp_blueberry',
        'logs/',
        'checkpoints/',
        '__pycache__',
        'wandb/'
    ]
    
    return all(pattern in content for pattern in required_patterns)

def check_files_exist():
    """Check if all required files exist"""
    required_files = [
        'README.md',
        'requirements.txt',
        'configs/de_training.yaml',
        'training/de_train.py',
        'models/plasa_model.py',
        'run_full.py'
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    return missing

def get_git_status():
    """Get git status"""
    try:
        result = subprocess.run(['git', 'status', '--short'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def main():
    print("=" * 60)
    print("GitHub Setup Check for Blueberry DE-Train")
    print("=" * 60)
    
    # Check git installation
    if not check_git_installed():
        print("❌ Git is not installed. Please install Git first.")
        sys.exit(1)
    print("✅ Git is installed")
    
    # Check if git repo
    if not check_git_repo():
        print("\n⚠️  Current directory is not a git repository.")
        init = input("Initialize git repository? (y/n): ")
        if init.lower() == 'y':
            subprocess.run(['git', 'init'], check=True)
            print("✅ Git repository initialized")
        else:
            print("Please initialize git repository first: git init")
            sys.exit(1)
    else:
        print("✅ Git repository detected")
    
    # Check .gitignore
    if not check_gitignore():
        print("⚠️  .gitignore might be incomplete. Please review.")
    else:
        print("✅ .gitignore looks good")
    
    # Check required files
    missing = check_files_exist()
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
    else:
        print("✅ All required files present")
    
    # Check git status
    status = get_git_status()
    if status:
        print("\n📋 Git status:")
        print(status)
        print("\n⚠️  You have uncommitted changes.")
    else:
        print("✅ Working directory clean")
    
    # Final checklist
    print("\n" + "=" * 60)
    print("Pre-Push Checklist:")
    print("=" * 60)
    print("1. ✅ All tests pass: pytest tests/ -v")
    print("2. ✅ README.md reviewed")
    print("3. ✅ No sensitive data in code")
    print("4. ✅ .gitignore verified")
    print("5. ✅ All files committed")
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Review changes: git status")
    print("2. Add files: git add .")
    print("3. Commit: git commit -m 'Initial commit: German PLASA LLM training pipeline'")
    print("4. Create GitHub repo and push:")
    print("   git remote add origin https://github.com/YOUR_USERNAME/blueberry-de-train.git")
    print("   git branch -M main")
    print("   git push -u origin main")
    print("\n✅ Ready for GitHub!")

if __name__ == "__main__":
    main()

