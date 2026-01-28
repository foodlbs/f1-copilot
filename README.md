# F1 RAG Microservices Platform

A production-ready Retrieval-Augmented Generation (RAG) system for Formula 1 data, featuring microservices architecture, multiple retrieval strategies, streaming responses, and comprehensive F1 data loading capabilities.

## 🏎️ What This Is

This project combines two powerful systems:

1. **RAG Microservices** - A sophisticated AI-powered chat system for querying F1 data
2. **F1 Data Loading** - Scripts to populate your vector database with comprehensive F1 historical data

## 🚀 Quick Start

### 1. Prerequisites

- Docker Desktop installed and running
- Pinecone account with API key
- OpenAI API key (for data loading)
- 8GB RAM minimum (16GB recommended)

### 2. Configure Environment

Create a `.env` file in the root directory:

```bash
# Pinecone (required for both)
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX=your-index-name

# OpenAI (required for data loading)
OPENAI_API_KEY=your-openai-api-key

# RAG System Configuration
OLLAMA_MODEL=llama3.1
OLLAMA_HOST=http://ollama:11434
REDIS_HOST=redis
REDIS_PORT=6379
API_URL=http://localhost:8000
```

### 3. Option A: Start RAG System (No Data Loading)

If you already have data in Pinecone or want to try the system first:

```bash
cd rag-microservices
./scripts/setup.sh
./scripts/start.sh
```

Open [http://localhost:3000](http://localhost:3000) to use the chat interface.

### 3. Option B: Load F1 Data First

To populate your Pinecone database with F1 historical data:

```bash
cd rag-microservices/data-loading

# Install dependencies
pip install -r requirements-data.txt

# Load complete F1 database (1950-latest)
python ingest_complete_database.py

# OR for quick test with modern data only (2020-latest)
python ingest_complete_database.py --modern-only
```

Then start the RAG system as shown in Option A.

## 📁 Project Structure

```
.
├── .env                          # Environment configuration
├── .env.example                  # Example configuration
├── data/                         # F1 CSV data files (optional)
│
└── rag-microservices/            # Main RAG system
    ├── README.md                 # Detailed RAG documentation
    ├── QUICKSTART.md             # Quick start guide
    ├── GET_STARTED.md            # 5-minute setup
    ├── PROJECT_SUMMARY.md        # Technical overview
    │
    ├── .env                      # RAG system config
    ├── docker-compose.yml        # Service orchestration
    │
    ├── services/                 # Microservices
    │   ├── rag-service/          # LangChain RAG service
    │   ├── kong/                 # API Gateway
    │   └── frontend/             # Next.js UI
    │
    ├── scripts/                  # Deployment scripts
    │   ├── setup.sh              # Initial setup
    │   ├── start.sh              # Start services
    │   ├── stop.sh               # Stop services
    │   ├── test-api.sh           # Test endpoints
    │   └── seed-data.sh          # Sample data
    │
    ├── docs/                     # Documentation
    │   ├── DEPLOYMENT.md         # Production deployment
    │   └── TROUBLESHOOTING.md    # Common issues
    │
    └── data-loading/             # F1 data loading scripts
        ├── README.md             # Data loading guide
        ├── ingest_complete_database.py
        ├── ingest_csv_data.py
        ├── ingest_fantasy_data.py
        ├── requirements-data.txt
        └── src/                  # Data loading modules
```

## 🎯 Features

### RAG Microservices

- **4 Retrieval Strategies**: Similarity, MMR, Multi-Query, Compression
- **Streaming Responses**: Real-time token-by-token generation
- **Conversation Memory**: Redis-backed session management
- **Kong API Gateway**: Rate limiting, CORS, health checks
- **Modern Frontend**: Next.js with TypeScript and Tailwind CSS
- **Local LLM**: Ollama (Llama 3.1) for complete privacy
- **Production Ready**: Docker containerized with health checks

### F1 Data Loading

- **Comprehensive Data**: 1950-present Formula 1 history
- **Multiple Sources**: CSV files, FastF1 API, Ergast API
- **Telemetry Data**: Modern races include detailed telemetry
- **Fantasy Metrics**: Driver value, points potential, recommendations
- **Flexible Loading**: Complete database or targeted updates

## 🔧 Common Use Cases

### Use Case 1: Chat About F1 History

1. Load historical data:
   ```bash
   cd rag-microservices/data-loading
   python ingest_complete_database.py
   ```

2. Start RAG system:
   ```bash
   cd ..
   ./scripts/start.sh
   ```

3. Ask questions like:
   - "Who won the 2023 Monaco Grand Prix?"
   - "What are Lewis Hamilton's career statistics?"
   - "Compare Max Verstappen and Charles Leclerc's performance at Monza"

### Use Case 2: F1 Fantasy Assistant

1. Load fantasy-optimized data:
   ```bash
   cd rag-microservices/data-loading
   python ingest_fantasy_data.py
   ```

2. Start RAG system and ask:
   - "Best driver picks for Silverstone?"
   - "Which drivers offer the best value this weekend?"
   - "Red Bull's historical performance at Spa?"

### Use Case 3: Load Custom Data

You can extend the data-loading scripts to ingest your own documents:

1. Add your custom data loader in `data-loading/src/`
2. Follow the pattern in `csv_data_ingestion.py`
3. Run your custom loader to populate Pinecone
4. Start the RAG system and chat with your data!

## 📊 Technology Stack

### RAG System
- **Backend**: Python 3.11, FastAPI, LangChain
- **LLM**: Ollama (Llama 3.1) - runs locally
- **Vector DB**: Pinecone (cloud-based)
- **Cache/Memory**: Redis
- **API Gateway**: Kong
- **Frontend**: Next.js 14, TypeScript, Tailwind CSS
- **Infrastructure**: Docker, Docker Compose

### Data Loading
- **Processing**: Python 3.11, pandas, numpy
- **APIs**: FastF1, Ergast, OpenF1
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector DB**: Pinecone

## 📖 Documentation

### Quick Start
- [GET_STARTED.md](rag-microservices/GET_STARTED.md) - 5-minute setup
- [QUICKSTART.md](rag-microservices/QUICKSTART.md) - Detailed guide

### Main Documentation
- [RAG System README](rag-microservices/README.md) - Complete RAG documentation
- [Data Loading README](rag-microservices/data-loading/README.md) - Data ingestion guide

### Technical Details
- [PROJECT_SUMMARY.md](rag-microservices/PROJECT_SUMMARY.md) - Architecture overview
- [DEPLOYMENT.md](rag-microservices/docs/DEPLOYMENT.md) - Production deployment
- [TROUBLESHOOTING.md](rag-microservices/docs/TROUBLESHOOTING.md) - Common issues

## 🌐 Service URLs

Once running, access these services:

- **Frontend**: [http://localhost:3000](http://localhost:3000)
- **Kong Gateway**: [http://localhost:8000](http://localhost:8000)
- **RAG Service**: [http://localhost:8001/health](http://localhost:8001/health)

## 🎨 Retrieval Strategies

The system supports 4 different retrieval strategies:

| Strategy | Best For | Speed |
|----------|----------|-------|
| **Similarity** | Direct questions | ⚡⚡⚡ Fast |
| **MMR** | Diverse perspectives | ⚡⚡ Medium |
| **Multi-Query** | Complex questions | ⚡ Slow |
| **Compression** | Long documents | ⚡ Slow |

Try them all using the frontend's strategy selector!

## 💰 Cost Considerations

### Data Loading (One-time)
- **Complete DB** (1950-latest): ~$10-12 in OpenAI embeddings
- **Modern only** (2020-latest): ~$3-4
- **Updates**: ~$1-2 per update

### Running RAG System
- **Ollama**: Free (runs locally)
- **Pinecone**: ~$70-100/month (Starter plan)
- **Hosting**: $50-200/month (if deploying to cloud)

## 🔐 Security Notes

This is a development/demo configuration. For production:

1. Add authentication to Kong (JWT/Key Auth plugins)
2. Use SSL/TLS with reverse proxy (Nginx + Let's Encrypt)
3. Secure environment variables (AWS Secrets Manager, etc.)
4. Implement proper network isolation
5. Enable audit logging

See [DEPLOYMENT.md](rag-microservices/docs/DEPLOYMENT.md) for details.

## 🚦 Architecture

```
┌─────────────┐
│   Browser   │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  Frontend   │  ← Next.js UI
│   :3000     │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│    Kong     │  ← API Gateway
│   :8000     │
└──────┬──────┘
       │
       ↓
   ┌──────┐
   │ RAG  │  ← LangChain Service
   │:8001 │
   └───┬──┘
       │
   ┌───┴──────────┐
   ↓              ↓
┌──────────┐  ┌─────────┐
│ Pinecone │  │ Ollama  │  ← Infrastructure
│  Vector  │  │ + Redis │     (Data + AI)
│   DB     │  │         │
└──────────┘  └─────────┘
       ↑
       │
   [Offline Data Loading]
   data-loading/ scripts
   populate Pinecone
```

## 🛠️ Development

### Running Tests

```bash
cd rag-microservices
./scripts/test-api.sh
```

### Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f rag-service
```

### Rebuilding After Changes

```bash
docker-compose up -d --build rag-service
```

## 🤝 Contributing

This is a complete, standalone project. Feel free to fork and customize for your needs!

## 📝 License

MIT License - use freely for your projects.

## 🆘 Need Help?

1. Check [TROUBLESHOOTING.md](rag-microservices/docs/TROUBLESHOOTING.md)
2. Review service logs: `docker-compose logs`
3. Verify environment configuration in `.env`
4. Ensure all services are healthy: `docker ps`

## 🎯 Next Steps

1. **Configure** your environment variables in `.env`
2. **Choose** your path:
   - Load F1 data → Use for F1 queries
   - Skip data → Use with your own documents
3. **Start** the RAG system: `cd rag-microservices && ./scripts/start.sh`
4. **Open** [http://localhost:3000](http://localhost:3000)
5. **Explore** different retrieval strategies
6. **Deploy** to production when ready

---

**Built with:** LangChain • Ollama • Pinecone • Kong • Next.js • FastAPI • Redis

**Ready to query your F1 knowledge base!** 🏎️🏆
