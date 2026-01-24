#!/bin/bash

set -e

echo "🚀 Starting RAG Microservices"
echo "============================"

# Start all services
echo "📦 Starting containers..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
echo "   This may take 2-3 minutes on first run..."

# Function to check service health
check_service() {
    local service=$1
    local url=$2
    local max_attempts=30
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            echo "   ✅ $service is ready"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done

    echo "   ❌ $service failed to start"
    return 1
}

# Check Redis
check_service "Redis" "http://localhost:6379"

# Check Ollama
check_service "Ollama" "http://localhost:11434/api/tags"

# Download Ollama model if not present
echo "📥 Checking Ollama model..."
if ! docker exec rag-ollama ollama list | grep -q "llama3.1"; then
    echo "   Downloading llama3.1 model (this will take a few minutes)..."
    docker exec rag-ollama ollama pull llama3.1
    echo "   ✅ Model downloaded"
else
    echo "   ✅ Model already present"
fi

# Check RAG Service
check_service "RAG Service" "http://localhost:8001/health"

# Check Ingestion Service
check_service "Ingestion Service" "http://localhost:8002/health"

# Check Kong
check_service "Kong Gateway" "http://localhost:8000/api/retrieval-strategies"

# Check Frontend
check_service "Frontend" "http://localhost:3000"

echo ""
echo "✅ All services are running!"
echo ""
echo "📊 Service URLs:"
echo "   Frontend:          http://localhost:3000"
echo "   Kong Proxy:        http://localhost:8000"
echo "   RAG Service:       http://localhost:8001"
echo "   Ingestion Service: http://localhost:8002"
echo "   Ollama:            http://localhost:11434"
echo ""
echo "📝 Try these commands:"
echo "   # Test chat"
echo "   curl -X POST http://localhost:8000/api/chat \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"query\": \"What is machine learning?\", \"retrieval_strategy\": \"similarity\"}'"
echo ""
echo "   # View logs"
echo "   docker-compose logs -f rag-service"
echo ""
echo "   # Stop services"
echo "   ./scripts/stop.sh"
