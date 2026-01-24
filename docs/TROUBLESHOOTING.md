# Troubleshooting Guide

Common issues and their solutions for the RAG Microservices platform.

## Table of Contents

1. [Startup Issues](#startup-issues)
2. [Service Connection Problems](#service-connection-problems)
3. [Ollama Issues](#ollama-issues)
4. [Pinecone Connection Issues](#pinecone-connection-issues)
5. [Frontend Issues](#frontend-issues)
6. [Performance Problems](#performance-problems)
7. [Docker Issues](#docker-issues)
8. [Kong Gateway Issues](#kong-gateway-issues)

## Startup Issues

### Services Won't Start

**Symptom**: `docker-compose up -d` fails

**Check**:
```bash
# View detailed logs
docker-compose logs

# Check specific service
docker-compose logs rag-service
```

**Common Causes**:

1. **Port conflicts**
   ```bash
   # Find what's using ports
   lsof -i :3000  # Frontend
   lsof -i :8000  # Kong
   lsof -i :8001  # RAG Service
   lsof -i :8002  # Ingestion

   # Kill conflicting process
   kill -9 <PID>
   ```

2. **Docker not running**
   ```bash
   # Check Docker status
   docker info

   # Start Docker Desktop (macOS)
   open -a Docker
   ```

3. **Insufficient resources**
   - Docker Desktop → Settings → Resources
   - Increase Memory to 8GB minimum
   - Increase Disk space to 20GB+

### Health Checks Failing

**Symptom**: Services start but health checks fail

**Solution**:
```bash
# Check health status
docker ps

# Manually test health endpoints
curl http://localhost:8001/health
curl http://localhost:8002/health

# Restart unhealthy service
docker-compose restart rag-service
```

### Environment Variables Not Loading

**Symptom**: "Environment variable not set" errors

**Solution**:
```bash
# Verify .env file exists
cat .env

# Load environment variables
export $(cat .env | xargs)

# Restart services
docker-compose down
docker-compose up -d
```

## Service Connection Problems

### RAG Service Can't Connect to Ollama

**Symptom**: "Connection refused" or "Ollama not reachable"

**Check**:
```bash
# Test Ollama directly
curl http://localhost:11434/api/tags

# Check if model exists
docker exec rag-ollama ollama list
```

**Solution**:
```bash
# Pull model manually
docker exec rag-ollama ollama pull llama3.1

# Restart RAG service
docker-compose restart rag-service
```

### Services Can't Communicate

**Symptom**: 502 Bad Gateway or connection timeouts

**Check network**:
```bash
# List networks
docker network ls

# Inspect network
docker network inspect rag-microservices_rag-network

# Check DNS resolution
docker exec rag-service ping ollama
docker exec rag-service ping redis
```

**Solution**:
```bash
# Recreate network
docker-compose down
docker-compose up -d
```

## Ollama Issues

### Model Download Stuck

**Symptom**: Ollama model download hangs at certain percentage

**Check progress**:
```bash
# View Ollama logs
docker logs rag-ollama -f
```

**Solution**:
```bash
# Cancel and retry
docker restart rag-ollama
docker exec rag-ollama ollama pull llama3.1

# If still failing, clear cache
docker-compose down -v
docker volume rm rag-microservices_ollama_data
docker-compose up -d
```

### Ollama Out of Memory

**Symptom**: OOM errors in Ollama logs

**Solution**:
```bash
# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory → 16GB

# Use smaller model
docker exec rag-ollama ollama pull llama2

# Update .env
OLLAMA_MODEL=llama2

# Restart services
docker-compose restart rag-service
```

### Slow Model Inference

**Symptom**: Responses take 30+ seconds

**Solutions**:

1. **Use smaller model**:
   ```bash
   OLLAMA_MODEL=mistral  # Faster than llama3.1
   ```

2. **Reduce context size** in `rag-service/app/main.py`:
   ```python
   retriever = vectorstore.as_retriever(
       search_kwargs={"k": 3}  # Reduce from 5 to 3
   )
   ```

3. **Enable GPU** (if available):
   ```yaml
   # docker-compose.yml
   ollama:
     deploy:
       resources:
         reservations:
           devices:
             - driver: nvidia
               count: 1
               capabilities: [gpu]
   ```

## Pinecone Connection Issues

### Authentication Failed

**Symptom**: "Invalid API key" or authentication errors

**Check**:
```bash
# Verify API key format
echo $PINECONE_API_KEY

# Test with curl
curl https://api.pinecone.io/indexes \
  -H "Api-Key: $PINECONE_API_KEY"
```

**Solution**:
```bash
# Update .env with correct key
PINECONE_API_KEY=pcsk_xxxxx_your_key_here

# Restart services
docker-compose restart rag-service ingestion-service
```

### Index Not Found

**Symptom**: "Index 'xyz' not found"

**Check**:
```bash
# List your indexes
# Visit: https://app.pinecone.io/

# Or use Python
docker exec rag-service python -c "
from pinecone import Pinecone
pc = Pinecone(api_key='$PINECONE_API_KEY')
print(pc.list_indexes())
"
```

**Solution**:
```bash
# Create index with correct dimensions
# Dimension for sentence-transformers/all-MiniLM-L6-v2: 384

# Update .env with correct index name
PINECONE_INDEX=your-actual-index-name

# Restart
docker-compose restart rag-service ingestion-service
```

### Dimension Mismatch

**Symptom**: "Dimension mismatch: expected 1536, got 384"

**Cause**: Index created with wrong dimension

**Solution**:
```bash
# Option 1: Create new index with dimension 384
# Visit Pinecone console and create index

# Option 2: Use different embedding model
# Edit services/rag-service/app/main.py
# Change embedding model to match your index dimension
```

### Rate Limiting

**Symptom**: "Rate limit exceeded" errors

**Solution**:
```bash
# Upgrade Pinecone plan
# Or reduce request frequency

# Add retry logic in services/rag-service/app/main.py:
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def query_pinecone():
    # Your query code
```

## Frontend Issues

### Frontend Not Loading

**Symptom**: Browser shows "Cannot connect" or blank page

**Check**:
```bash
# Test frontend directly
curl http://localhost:3000

# Check logs
docker-compose logs frontend
```

**Solution**:
```bash
# Rebuild frontend
docker-compose up -d --build frontend

# Check if port 3000 is available
lsof -i :3000
```

### CORS Errors in Browser

**Symptom**: Console shows CORS policy errors

**Check Kong CORS configuration**:
```bash
# Verify CORS settings
cat services/kong/kong.yml | grep -A 10 cors
```

**Solution**:
Already configured. If still issues:
```yaml
# services/kong/kong.yml
plugins:
  - name: cors
    config:
      origins:
        - "http://localhost:3000"
        - "https://yourdomain.com"
      methods:
        - GET
        - POST
        - DELETE
        - OPTIONS
      headers:
        - "*"
      credentials: true
      max_age: 3600
```

### Streaming Not Working

**Symptom**: Messages appear all at once instead of streaming

**Check**:
```bash
# Test streaming directly
curl -N -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"query": "test", "stream": true}'
```

**Solution**:

1. **Browser caching**: Hard refresh (Cmd/Ctrl + Shift + R)
2. **Proxy buffering**: If using Nginx, add:
   ```nginx
   proxy_buffering off;
   proxy_cache off;
   ```

### Session Not Persisting

**Symptom**: Conversation history lost after each message

**Check Redis**:
```bash
# Test Redis connection
docker exec rag-redis redis-cli ping

# Check stored sessions
docker exec rag-redis redis-cli KEYS "chat_history:*"
```

**Solution**:
```bash
# Restart Redis
docker-compose restart redis

# Verify connection in RAG service
docker exec rag-service python -c "
import redis
r = redis.Redis(host='redis', port=6379)
print(r.ping())
"
```

## Performance Problems

### Slow Query Response

**Symptom**: Queries take 10+ seconds

**Diagnosis**:
```bash
# Check service resource usage
docker stats

# Check Ollama performance
docker logs rag-ollama | grep -i "time"
```

**Solutions**:

1. **Reduce top_k**:
   ```python
   # In frontend
   top_k: 3  # Instead of 5
   ```

2. **Use faster strategy**:
   ```javascript
   retrieval_strategy: "similarity"  // Fastest
   ```

3. **Optimize Pinecone queries**:
   ```python
   # Add metadata filters to narrow search
   retriever = vectorstore.as_retriever(
       search_kwargs={
           "k": 5,
           "filter": {"source": "specific_source"}
       }
   )
   ```

### High Memory Usage

**Symptom**: Services using excessive RAM

**Check**:
```bash
docker stats

# Expected usage:
# - Ollama: 4-8GB (model size)
# - RAG Service: 1-2GB
# - Ingestion: 500MB-1GB
# - Redis: 50-100MB
# - Frontend: 100-200MB
```

**Solution**:
```bash
# Restart services to clear memory
docker-compose restart

# Limit container memory in docker-compose.yml:
services:
  rag-service:
    deploy:
      resources:
        limits:
          memory: 2G
```

### Redis Memory Full

**Symptom**: "OOM command not allowed" errors

**Solution**:
```bash
# Clear old sessions
docker exec rag-redis redis-cli FLUSHALL

# Configure maxmemory
docker exec rag-redis redis-cli CONFIG SET maxmemory 512mb
docker exec rag-redis redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

## Docker Issues

### Disk Space Full

**Symptom**: "no space left on device"

**Solution**:
```bash
# Clean up Docker
docker system prune -a --volumes

# Remove unused images
docker image prune -a

# Remove stopped containers
docker container prune

# Check disk usage
docker system df
```

### Build Failures

**Symptom**: Docker build errors

**Common causes**:

1. **Network issues**:
   ```bash
   # Retry with no cache
   docker-compose build --no-cache
   ```

2. **Dependency conflicts**:
   ```bash
   # Update requirements.txt
   # Pin specific versions
   ```

3. **Permission errors**:
   ```bash
   # Run with sudo (Linux)
   sudo docker-compose build
   ```

## Kong Gateway Issues

### Kong Not Starting

**Symptom**: Kong container exits immediately

**Check**:
```bash
# View Kong logs
docker logs rag-kong

# Common error: Invalid kong.yml
docker exec rag-kong kong check /usr/local/kong/declarative/kong.yml
```

**Solution**:
```bash
# Validate kong.yml syntax
# Fix any YAML errors

# Restart Kong
docker-compose restart kong
```

### Routes Not Working

**Symptom**: 404 errors on API calls

**Check routes**:
```bash
# List Kong routes
curl http://localhost:8444/routes
```

**Solution**:
```bash
# Verify kong.yml routes section
cat services/kong/kong.yml

# Ensure strip_path is set correctly
strip_path: false  # Don't remove /api prefix
```

### Rate Limiting Too Aggressive

**Symptom**: "API rate limit exceeded" errors

**Solution**:
```yaml
# Increase limits in services/kong/kong.yml
plugins:
  - name: rate-limiting
    config:
      second: 20    # Increase from 10
      minute: 200   # Increase from 100
      hour: 2000    # Increase from 1000
```

## Getting More Help

### Enable Debug Logging

**RAG Service**:
```python
# In services/rag-service/app/main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Kong**:
```yaml
environment:
  - KONG_LOG_LEVEL=debug
```

### Collect Diagnostic Info

```bash
#!/bin/bash
# diagnose.sh

echo "=== System Info ==="
docker version
docker-compose version

echo "\n=== Running Containers ==="
docker ps -a

echo "\n=== Service Logs ==="
docker-compose logs --tail=50

echo "\n=== Resource Usage ==="
docker stats --no-stream

echo "\n=== Network Info ==="
docker network inspect rag-microservices_rag-network

echo "\n=== Volume Info ==="
docker volume ls
docker volume inspect rag-microservices_redis_data
docker volume inspect rag-microservices_ollama_data
```

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| "Connection refused" | Service not running | Start service: `docker-compose up -d` |
| "401 Unauthorized" | Invalid API key | Check `.env` file |
| "404 Not Found" | Wrong endpoint/route | Verify URL path |
| "500 Internal Server Error" | Service error | Check logs: `docker-compose logs` |
| "503 Service Unavailable" | Service unhealthy | Wait for health checks or restart |
| "OOM killed" | Out of memory | Increase Docker memory limit |

### Still Need Help?

1. Check all logs: `docker-compose logs > debug.log`
2. Review `.env` configuration
3. Verify Pinecone credentials
4. Test each service individually
5. Check GitHub issues (if using from repo)

---

For more information, see [README.md](../README.md) and [DEPLOYMENT.md](DEPLOYMENT.md).
