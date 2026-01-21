#!/bin/bash

# F1 Knowledge Base Setup Script
# Automated setup for the F1 race analysis system

set -e  # Exit on error

echo "=========================================="
echo "F1 Knowledge Base Setup"
echo "=========================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"

# Create virtual environment
echo ""
echo "🔧 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "   ✓ Virtual environment created"
else
    echo "   ℹ️  Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
echo "   (This may take a few minutes...)"
pip install -r requirements.txt --quiet
echo "   ✓ Dependencies installed"

# Create directories
echo ""
echo "📁 Creating directories..."
mkdir -p cache/f1_data
mkdir -p cache/fastf1
mkdir -p data
mkdir -p logs
mkdir -p src
echo "   ✓ Directories created"

# Check for .env file
echo ""
echo "🔑 Checking environment configuration..."
if [ ! -f ".env" ]; then
    echo "   ⚠️  .env file not found"
    echo "   Creating .env from template..."
    cp .env.example .env
    echo ""
    echo "   ⚠️  IMPORTANT: Edit .env file with your API keys!"
    echo "   You need:"
    echo "      • PINECONE_API_KEY from https://www.pinecone.io/"
    echo "      • OPENAI_API_KEY from https://platform.openai.com/"
else
    echo "   ✓ .env file exists"
fi

# Test configuration
echo ""
echo "🧪 Testing configuration..."
python3 config.py
config_status=$?

if [ $config_status -eq 0 ]; then
    echo "   ✓ Configuration valid"
else
    echo "   ❌ Configuration invalid - check your .env file"
fi

# Summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Verify API keys in .env file"
echo ""
echo "3. Build knowledge base:"
echo "   cd src && python knowledge_base_builder.py"
echo ""
echo "4. Or run examples:"
echo "   python example_usage.py"
echo ""
echo "5. Read documentation:"
echo "   cat README.md"
echo "   cat QUICKSTART.md"
echo ""
echo "=========================================="
echo "Happy racing! 🏎️"
echo "=========================================="
