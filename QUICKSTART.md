# Quick Start Guide - RAG Microservices

Get your RAG system running in minutes!

## Prerequisites Checklist

- [ ] Docker Desktop installed and running
- [ ] Pinecone account created
- [ ] Pinecone API key obtained
- [ ] Pinecone index created (dimension: 384 for sentence-transformers/all-MiniLM-L6-v2)
- [ ] At least 8GB RAM available
- [ ] 20GB free disk space

## Step-by-Step Setup

### 1. Configure Your Environment

Edit the `.env` file and add your Pinecone credentials:

```bash
PINECONE_API_KEY=pcsk_xxxxx_your_actual_key_here
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX=f1-race-knowledge
```

### 2. Run Setup Script

```bash
cd rag-microservices
./scripts/setup.sh
```

This checks prerequisites and builds all Docker images.

### 3. Start All Services

```bash
./scripts/start.sh
```

**First run takes 10-15 minutes** to download:
- Ollama Llama 3.1 model (~4GB)
- Python dependencies with embedding models
- Node modules

Subsequent starts take 30-60 seconds.

### 4. Seed Sample Data

In a new terminal:

```bash
./scripts/seed-data.sh
```

This adds 10 documents about machine learning to your vector database.

### 5. Open the Application

Visit [http://localhost:3000](http://localhost:3000)

## Try It Out

### Example Questions (after seeding data)

1. "What is machine learning?"
2. "Explain supervised learning"
3. "What is the difference between supervised and unsupervised learning?"
4. "Tell me about deep learning"
5. "What is overfitting and how do I prevent it?"

### Test Different Retrieval Strategies

In the frontend:
1. Type a question in the input box
2. Select a retrieval strategy from the dropdown
3. Click "Test Strategies" to compare all strategies
4. Enable/disable streaming to see the difference

## Verify Everything Works

Run the test script:

```bash
./scripts/test-api.sh
```

This tests:
- Health endpoints
- Chat functionality
- Document ingestion
- Database statistics

## Common Issues

### "Port already in use"

Stop conflicting services or edit `docker-compose.yml` to use different ports.

### "Cannot connect to Docker"

Ensure Docker Desktop is running.

### "Pinecone connection failed"

Verify your API key and index name in `.env`.

### Ollama model stuck downloading

Check progress:
```bash
docker logs rag-ollama -f
```

Manually pull:
```bash
docker exec rag-ollama ollama pull llama3.1
```

## Next Steps

### Ingest Your Own Documents

**PDF:**
```bash
curl -X POST http://localhost:8000/api/ingest/pdf \
  -F "file=@/path/to/your/document.pdf"
```

**Text:**
```bash
curl -X POST http://localhost:8000/api/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "documents": [
      {"content": "Your text here", "metadata": {"source": "custom"}}
    ]
  }'
```

### Monitor Your System

```bash
# View logs
docker-compose logs -f

# Check specific service
docker-compose logs -f rag-service

# Monitor resource usage
docker stats
```

### Stop the System

```bash
./scripts/stop.sh
```

## Architecture at a Glance

```
┌─────────────┐
│   Browser   │
│ :3000       │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│    Kong     │  ← API Gateway (rate limiting, routing)
│ :8000       │
└──────┬──────┘
       │
       ↓
   ┌──────┐
   │ RAG  │
   │:8001 │
   └───┬──┘
       │
       ↓
   ┌─────────────┐
   │   Pinecone  │  ← Your vector DB
   │   Ollama    │  ← Local LLM
   │   Redis     │  ← Conversation memory
   └─────────────┘
```

## What You Get

- **Frontend**: Modern chat interface with streaming
- **RAG Service**: 4 retrieval strategies + conversation memory
- **Kong Gateway**: Rate limiting, CORS, health checks
- **Ollama**: Local Llama 3.1 inference
- **Pinecone**: Vector similarity search
- **Redis**: Session and conversation storage
- **Data Loading**: Offline scripts to populate Pinecone with F1 data

## Configuration

All configuration is in `.env`. Key settings:

- `OLLAMA_MODEL`: Change to llama2, mistral, etc.
- `PINECONE_INDEX`: Your index name
- Rate limits: Edit `services/kong/kong.yml`
- Data loading: See `data-loading/README.md`

## Need Help?

1. Check `README.md` for detailed documentation
2. Review `docker-compose logs` for errors
3. Verify `.env` configuration
4. Ensure Docker has enough resources (Docker Desktop → Settings → Resources)

---

**Ready to build amazing RAG applications!**
