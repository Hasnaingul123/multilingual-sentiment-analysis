#!/bin/bash
#
# Setup script for Multilingual Sentiment Analysis System
# This script creates a virtual environment and installs dependencies
#

set -e  # Exit on error

echo "========================================="
echo "Multilingual Sentiment Analysis Setup"
echo "========================================="
echo ""

# Detect OS
OS="$(uname)"
echo "Detected OS: $OS"
echo ""

# Python version check
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "Checking Python version..."
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check if Python version is >= 3.8
REQUIRED_VERSION="3.8"
if ! $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "ERROR: Python 3.8+ is required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "✓ Python version check passed"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    $PYTHON_CMD -m venv venv
    echo "✓ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
if [ "$OS" = "Windows_NT" ]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Install package in editable mode
echo "Installing package in editable mode..."
pip install -e .
echo "✓ Package installed"
echo ""

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p logs checkpoints data/{raw,processed,cache}
echo "✓ Directories created"
echo ""

# Run verification
echo "Running verification tests..."
python3 -c "
import sys
sys.path.insert(0, 'src')

# Test imports
try:
    from utils import load_config, setup_logging
    print('✓ Utils import successful')
    
    # Test config loading
    config = load_config('config/model_config.yaml')
    print('✓ Config loading successful')
    
    # Test logger
    logger = setup_logging(log_dir='logs')
    logger.info('Setup verification test')
    print('✓ Logger test successful')
    
    print('')
    print('========================================')
    print('Setup completed successfully!')
    print('========================================')
    print('')
    print('To activate the virtual environment:')
    if '$OS' == 'Windows_NT':
        print('  venv\\\\Scripts\\\\activate')
    else:
        print('  source venv/bin/activate')
    print('')
    print('To run tests:')
    print('  pytest tests/')
    print('')
    
except Exception as e:
    print(f'✗ Setup verification failed: {e}')
    sys.exit(1)
"

echo "Setup script completed!"
