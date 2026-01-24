#!/bin/bash

set -e

echo "🚀 RAG Microservices Setup"
echo "=========================="

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose."
    exit 1
fi

echo "✅ Docker and Docker Compose found"

# Check .env file
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "📝 Please create .env file with your Pinecone credentials:"
    echo ""
    cat << 'EOF'
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX=your-index-name
OLLAMA_MODEL=llama3.1
OLLAMA_HOST=http://ollama:11434
REDIS_HOST=redis
REDIS_PORT=6379
API_URL=http://localhost:8000
EOF
    exit 1
fi

echo "✅ .env file found"

# Load environment variables
source .env

# Validate required variables
if [ -z "$PINECONE_API_KEY" ] || [ -z "$PINECONE_INDEX" ]; then
    echo "❌ Missing required environment variables in .env"
    exit 1
fi

echo "✅ Environment variables loaded"

# Pull base images
echo "📥 Pulling Docker base images..."
docker pull python:3.11-slim
docker pull node:20-alpine
docker pull kong:3.5
docker pull redis:7-alpine
docker pull ollama/ollama:latest

echo "✅ Base images pulled"

# Build custom images
echo "🔨 Building custom images..."
docker-compose build

echo "✅ Images built successfully"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Run './scripts/start.sh' to start all services"
echo "2. Wait for Ollama to download the model (first time only)"
echo "3. Open http://localhost:3000 in your browser"
