# Quick Reference Guide

## 🚀 Getting Started (Choose Your Path)

### Path 1: RAG System with F1 Data

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with your keys (Pinecone + OpenAI)

# 2. Load F1 data
cd rag-microservices/data-loading
pip install -r requirements-data.txt
python ingest_complete_database.py --modern-only  # Fast test

# 3. Start RAG system
cd ..
./scripts/start.sh

# 4. Open browser
open http://localhost:3000
```

### Path 2: RAG System with Your Documents

```bash
# 1. Configure environment
cd rag-microservices
cp .env.example .env
# Edit .env with your Pinecone key (no OpenAI needed)

# 2. Start RAG system
./scripts/setup.sh
./scripts/start.sh

# 3. Upload documents via API or UI
curl -X POST http://localhost:8000/api/ingest/pdf \
  -F "file=@your-document.pdf"

# 4. Open browser
open http://localhost:3000
```

## 📁 Where Everything Is

```
BuildWatch/
├── README.md                          # Start here
├── .env                               # Your config
├── data/                              # F1 CSV files
└── rag-microservices/                 # Main system
    ├── scripts/start.sh               # Start everything
    ├── services/                      # Microservices code
    └── data-loading/                  # F1 data scripts
        └── ingest_complete_database.py
```

## 🛠 Common Commands

### RAG System
```bash
cd rag-microservices

# Start services
./scripts/start.sh

# Stop services
./scripts/stop.sh

# Test API
./scripts/test-api.sh

# View logs
docker-compose logs -f

# Restart a service
docker-compose restart rag-service
```

### Data Loading
```bash
cd rag-microservices/data-loading

# Complete database (1950-latest)
python ingest_complete_database.py

# Modern data only (2020-latest) - Fast test
python ingest_complete_database.py --modern-only

# Update with latest races
python ingest_complete_database.py --update-only

# CSV data only
python ingest_csv_data.py

# Fantasy data
python ingest_fantasy_data.py
```

## 🔗 Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Frontend | http://localhost:3000 | Chat interface |
| Kong Gateway | http://localhost:8000 | API proxy |
| RAG Service | http://localhost:8001/health | LangChain service |
| Ingestion | http://localhost:8002/health | Document processing |

## 📖 Documentation Quick Links

| Document | When to Read |
|----------|--------------|
| [README.md](README.md) | Project overview & getting started |
| [rag-microservices/GET_STARTED.md](rag-microservices/GET_STARTED.md) | 5-minute quick start |
| [rag-microservices/README.md](rag-microservices/README.md) | Complete RAG documentation |
| [data-loading/README.md](rag-microservices/data-loading/README.md) | Data loading guide |
| [DEPLOYMENT.md](rag-microservices/docs/DEPLOYMENT.md) | Production deployment |
| [TROUBLESHOOTING.md](rag-microservices/docs/TROUBLESHOOTING.md) | Having issues? |

## ⚙️ Configuration

### Required Environment Variables

```bash
# .env (root or rag-microservices/)

# For RAG System (required)
PINECONE_API_KEY=your-key
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX=your-index-name

# For Data Loading (required)
OPENAI_API_KEY=your-openai-key

# Optional (have defaults)
OLLAMA_MODEL=llama3.1
REDIS_HOST=redis
API_URL=http://localhost:8000
```

## 🎯 Retrieval Strategies

Try different strategies in the UI:

| Strategy | Best For | Speed |
|----------|----------|-------|
| Similarity | Direct questions | ⚡⚡⚡ |
| MMR | Diverse answers | ⚡⚡ |
| Multi-Query | Complex questions | ⚡ |
| Compression | Long documents | ⚡ |

## 💡 Example Questions

### F1 Data (if loaded)
- "Who won the 2023 Monaco Grand Prix?"
- "What are Lewis Hamilton's career stats?"
- "Compare Max Verstappen and Charles Leclerc"
- "Best drivers for Silverstone this weekend?"

### Your Documents
- "Summarize the main points"
- "What does the contract say about..."
- "List all requirements mentioned"

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Port already in use | Stop conflicting service or change ports |
| Cannot connect to Docker | Start Docker Desktop |
| Pinecone auth failed | Check API key in .env |
| Services not starting | Run `docker-compose logs` |
| Ollama slow | First time downloads model (5-10 min) |

## 💰 Costs

### One-Time (Data Loading)
- Complete DB: ~$10-12
- Modern only: ~$3-4
- Updates: ~$1-2

### Monthly (Running System)
- Pinecone: ~$70-100
- Ollama: Free (local)
- Cloud hosting: $50-200 (optional)

## 📊 Architecture Overview

```
Browser → Kong Gateway → RAG Service → Ollama (LLM)
                      → Ingestion     → Pinecone (Vectors)
                                      → Redis (Memory)
```

## 🔐 Security Checklist

For production:
- [ ] Add Kong authentication (JWT/Key Auth)
- [ ] Use SSL/TLS with reverse proxy
- [ ] Secure environment variables
- [ ] Enable network isolation
- [ ] Set up audit logging

## 🎓 Learning Path

1. **Day 1**: Follow GET_STARTED.md, start system, try basic queries
2. **Day 2**: Load F1 data or your documents, test retrieval strategies
3. **Day 3**: Read PROJECT_SUMMARY.md, understand architecture
4. **Day 4**: Customize for your use case
5. **Day 5**: Deploy to production with DEPLOYMENT.md

## 🆘 Need Help?

1. Check [TROUBLESHOOTING.md](rag-microservices/docs/TROUBLESHOOTING.md)
2. Review logs: `docker-compose logs`
3. Verify .env configuration
4. Check service health: `docker ps`

## ✅ Quick Health Check

```bash
# All services healthy?
docker ps

# Test RAG service
curl http://localhost:8001/health

# Test Kong gateway
curl http://localhost:8000/api/retrieval-strategies

# Test frontend
open http://localhost:3000
```

## 🚀 Production Deployment

```bash
# 1. Update .env for production
# 2. Build for production
docker-compose -f docker-compose.prod.yml build

# 3. Start
docker-compose -f docker-compose.prod.yml up -d

# 4. Set up reverse proxy (Nginx)
# 5. Configure SSL (Let's Encrypt)
# 6. Enable monitoring

# See DEPLOYMENT.md for details
```

---

**Quick tip**: Bookmark this page for fast reference!
