#!/bin/bash

echo "🧪 Testing RAG API"
echo "================="

# Test health endpoints
echo ""
echo "1️⃣ Testing Health Endpoints"
echo "----------------------------"

echo "RAG Service:"
curl -s http://localhost:8001/health | jq

echo ""
echo "Ingestion Service:"
curl -s http://localhost:8002/health | jq

echo ""
echo "Kong Gateway:"
curl -s http://localhost:8000/api/retrieval-strategies | jq

# Test chat endpoint
echo ""
echo "2️⃣ Testing Chat Endpoint"
echo "------------------------"

curl -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What is machine learning?",
    "retrieval_strategy": "similarity",
    "stream": false,
    "top_k": 3
  }' | jq

# Test ingestion
echo ""
echo "3️⃣ Testing Ingestion"
echo "--------------------"

curl -X POST http://localhost:8000/api/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "documents": [
      {
        "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
        "metadata": {"source": "test", "topic": "ML"}
      }
    ]
  }' | jq

# Get stats
echo ""
echo "4️⃣ Database Statistics"
echo "----------------------"

curl -s http://localhost:8000/api/stats | jq

echo ""
echo "✅ Tests complete!"
