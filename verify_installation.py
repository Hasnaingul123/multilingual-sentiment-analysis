#!/usr/bin/env python3
"""
Installation Verification Script

Checks that all components are properly installed and configured.
"""

import sys
import os
from pathlib import Path
import importlib.util

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def print_status(message, status='info'):
    """Print colored status message."""
    if status == 'success':
        print(f"{GREEN}✓{RESET} {message}")
    elif status == 'error':
        print(f"{RED}✗{RESET} {message}")
    elif status == 'warning':
        print(f"{YELLOW}⚠{RESET} {message}")
    else:
        print(f"  {message}")

def check_python_version():
    """Check Python version."""
    print("\n1. Checking Python version...")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version >= (3, 8):
        print_status(f"Python {version_str} (>= 3.8 required)", 'success')
        return True
    else:
        print_status(f"Python {version_str} (3.8+ required)", 'error')
        return False

def check_dependencies():
    """Check if required packages are installed."""
    print("\n2. Checking dependencies...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('yaml', 'PyYAML'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'scikit-learn'),
    ]
    
    all_installed = True
    for package, name in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is not None:
            print_status(f"{name} installed", 'success')
        else:
            print_status(f"{name} not found", 'error')
            all_installed = False
    
    return all_installed

def check_project_structure():
    """Check if project directories exist."""
    print("\n3. Checking project structure...")
    
    required_dirs = [
        'config',
        'src',
        'src/utils',
        'src/data',
        'src/models',
        'src/preprocessing',
        'src/training',
        'src/evaluation',
        'src/inference',
        'tests',
        'logs',
        'checkpoints',
        'data/raw',
        'data/processed',
        'data/cache',
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print_status(f"{dir_path}/ exists", 'success')
        else:
            print_status(f"{dir_path}/ missing", 'error')
            all_exist = False
    
    return all_exist

def check_config_files():
    """Check if configuration files exist."""
    print("\n4. Checking configuration files...")
    
    config_files = [
        'config/model_config.yaml',
        'config/training_config.yaml',
        'config/preprocessing_config.yaml',
    ]
    
    all_exist = True
    for config_file in config_files:
        if Path(config_file).exists():
            print_status(f"{config_file} exists", 'success')
        else:
            print_status(f"{config_file} missing", 'error')
            all_exist = False
    
    return all_exist

def check_module_imports():
    """Check if custom modules can be imported."""
    print("\n5. Checking module imports...")
    
    # Add src to path
    sys.path.insert(0, 'src')
    
    modules = [
        ('utils.config_loader', 'ConfigLoader'),
        ('utils.logger', 'Logger'),
    ]
    
    all_imported = True
    for module_name, class_name in modules:
        try:
            module = importlib.import_module(module_name)
            getattr(module, class_name)
            print_status(f"{module_name}.{class_name} imported", 'success')
        except Exception as e:
            print_status(f"{module_name}.{class_name} import failed: {e}", 'error')
            all_imported = False
    
    return all_imported

def test_config_loading():
    """Test configuration loading."""
    print("\n6. Testing configuration loading...")
    
    sys.path.insert(0, 'src')
    
    try:
        from utils.config_loader import load_config
        
        # Test loading model config
        config = load_config('config/model_config.yaml')
        print_status("Model config loaded successfully", 'success')
        
        # Verify key fields
        assert 'model' in config
        assert 'multitask' in config
        print_status("Config structure validated", 'success')
        
        return True
    except Exception as e:
        print_status(f"Config loading failed: {e}", 'error')
        return False

def test_logger():
    """Test logger functionality."""
    print("\n7. Testing logger...")
    
    sys.path.insert(0, 'src')
    
    try:
        from utils.logger import setup_logging
        
        logger = setup_logging(log_dir='logs')
        logger.info("Verification test log message")
        print_status("Logger initialized successfully", 'success')
        
        # Check if log file was created
        log_files = list(Path('logs').glob('*.log'))
        if log_files:
            print_status(f"Log file created: {log_files[-1].name}", 'success')
        
        return True
    except Exception as e:
        print_status(f"Logger test failed: {e}", 'error')
        return False

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Multilingual Sentiment Analysis - Installation Verification")
    print("=" * 60)
    
    checks = [
        check_python_version,
        check_dependencies,
        check_project_structure,
        check_config_files,
        check_module_imports,
        test_config_loading,
        test_logger,
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print_status(f"Check failed with exception: {e}", 'error')
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if all(results):
        print_status(f"All checks passed ({passed}/{total})", 'success')
        print("\n✓ System is ready for development!")
        print("\nNext steps:")
        print("  1. Review configuration files in config/")
        print("  2. Prepare your dataset in data/raw/")
        print("  3. Proceed to Phase 2: Data Pipeline Implementation")
        return 0
    else:
        print_status(f"{passed}/{total} checks passed", 'warning')
        print("\n⚠ Please fix the errors above before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
