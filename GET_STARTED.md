# Get Started in 5 Minutes

## Before You Begin

You need:
1. **Docker Desktop** installed and running
2. **Pinecone account** with:
   - API key
   - An index created with dimension **384**
3. At least **8GB RAM** available

## Setup Steps

### 1. Configure Environment (1 minute)

Open `.env` and update these three lines with YOUR credentials:

```bash
PINECONE_API_KEY=YOUR_ACTUAL_API_KEY_HERE
PINECONE_ENVIRONMENT=YOUR_ENVIRONMENT_HERE  # e.g., us-east-1-aws
PINECONE_INDEX=YOUR_INDEX_NAME_HERE
```

**How to get these**:
- Login to [Pinecone Console](https://app.pinecone.io/)
- Copy your API key from the dashboard
- Note your environment (shown next to the API key)
- Use an existing index or create one with dimension 384

### 2. Run Setup (2 minutes)

```bash
./scripts/setup.sh
```

This will:
- Check Docker is running
- Verify your `.env` file
- Pull Docker images
- Build custom services

### 3. Start Everything (5-10 minutes first time)

```bash
./scripts/start.sh
```

**First run**: Downloads Llama 3.1 model (~4GB). Go grab a coffee!

**Subsequent runs**: Takes 30-60 seconds

### 4. Add Sample Data (30 seconds)

In a new terminal:

```bash
./scripts/seed-data.sh
```

This adds 10 machine learning documents to test with.

### 5. Open Your Browser

Visit: [http://localhost:3000](http://localhost:3000)

## Try These Questions

After seeding data:

1. "What is machine learning?"
2. "Explain supervised learning"
3. "What's the difference between supervised and unsupervised learning?"
4. "Tell me about deep learning"
5. "What is overfitting?"

## Experiment with Strategies

Try each retrieval strategy to see the differences:

- **Similarity** (Fast) - Direct answers
- **MMR** (Medium) - Diverse perspectives
- **Multi-Query** (Slow) - Comprehensive results
- **Compression** (Slow) - Focused extraction

Click "Test Strategies" to compare all at once!

## What's Running?

After startup, you have:

| Service | URL | Purpose |
|---------|-----|---------|
| Frontend | [http://localhost:3000](http://localhost:3000) | Chat UI |
| Kong Gateway | [http://localhost:8000](http://localhost:8000) | API Gateway |
| RAG Service | [http://localhost:8001/health](http://localhost:8001/health) | LangChain RAG |

## Common First-Time Issues

### "Port already in use"

Something is using port 3000, 8000, or 8001.

**Fix**: Stop the conflicting service or edit `docker-compose.yml`

### "Cannot connect to Docker"

Docker Desktop isn't running.

**Fix**: Open Docker Desktop and wait for it to start

### "Pinecone authentication failed"

Your API key is incorrect.

**Fix**: Double-check your API key in `.env`

### "Index not found"

Index name is wrong or doesn't exist.

**Fix**: Verify index name in Pinecone console and update `.env`

### Services stuck "starting"

Ollama is downloading the model (first time only).

**Check progress**:
```bash
docker logs rag-ollama -f
```

**Wait**: Can take 5-10 minutes for 4GB download

## Next Steps

### Add Your Own Documents

**Via API**:
```bash
curl -X POST http://localhost:8000/api/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "documents": [{
      "content": "Your text here",
      "metadata": {"source": "custom"}
    }]
  }'
```

**Upload PDF**:
```bash
curl -X POST http://localhost:8000/api/ingest/pdf \
  -F "file=@/path/to/your.pdf"
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f rag-service
```

### Stop Everything

```bash
./scripts/stop.sh
```

### Start Again

```bash
./scripts/start.sh
```

(Much faster second time - no model download!)

## Help & Documentation

- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)
- **Full Documentation**: [README.md](README.md)
- **Deployment Guide**: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
- **Troubleshooting**: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- **Project Overview**: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

## Verify Everything Works

Run automated tests:

```bash
./scripts/test-api.sh
```

Should see all green checkmarks ✅

## Architecture

```
You → Frontend (Next.js) → Kong Gateway → RAG Service (LangChain)
                                            ↓
                              Ollama + Pinecone + Redis
                                            ↑
                                   [Offline data loading]
                                     data-loading/ scripts
```

## What You Can Do

- ✅ Chat with your F1 data
- ✅ Compare retrieval strategies
- ✅ Load comprehensive F1 historical data
- ✅ Maintain conversation history
- ✅ Stream responses in real-time
- ✅ Scale for production use

## Important Notes

- **First startup**: 5-10 minutes (model download)
- **Subsequent startups**: 30-60 seconds
- **Memory needed**: 8GB minimum, 16GB recommended
- **Disk space**: 20GB for images and models

## You're Ready!

Your production-ready RAG system is now running. Start chatting at:

🚀 [http://localhost:3000](http://localhost:3000)

---

**Need help?** Check the troubleshooting guide or review the logs.
