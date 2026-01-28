#!/bin/bash

echo "🛑 Stopping RAG Microservices"
echo "============================"

docker-compose down

echo "✅ All services stopped"
echo ""
echo "💡 To remove volumes (will delete Redis data):"
echo "   docker-compose down -v"
