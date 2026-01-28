# F1 RAG System - Complete Guide

**Production-ready AI-powered chat system for Formula 1 data with microservices architecture**

## 📋 Table of Contents

1. [What Is This?](#what-is-this)
2. [Quick Start (5 Minutes)](#quick-start-5-minutes)
3. [System Architecture](#system-architecture)
4. [Setup Instructions](#setup-instructions)
5. [Loading F1 Data](#loading-f1-data)
6. [Using the System](#using-the-system)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)
9. [Deployment](#deployment)
10. [Cost & Performance](#cost--performance)

---

## What Is This?

An intelligent chat system that lets you ask questions about Formula 1 using natural language. The system combines:

- **AI-Powered Chat**: Ask questions like "Who won Monaco 2023?" and get instant answers
- **Multiple Search Strategies**: 4 different ways to find information (Similarity, MMR, Multi-Query, Compression)
- **Comprehensive F1 Data**: Complete history from 1950 to present, including telemetry and fantasy stats
- **Streaming Responses**: See answers appear in real-time
- **Modern Architecture**: Production-ready microservices with Docker

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Next.js 14 + TypeScript | Chat interface |
| **API Gateway** | Kong | Request routing, rate limiting, CORS |
| **RAG Service** | FastAPI + LangChain | Core chat & retrieval logic |
| **LLM** | OpenAI GPT-4o-mini (or Ollama) | Natural language understanding |
| **Vector DB** | Pinecone | Document search & retrieval |
| **Memory** | Redis | Conversation history |
| **Embeddings** | OpenAI text-embedding-3-large | Vector embeddings |

---

## Quick Start (5 Minutes)

### Prerequisites

- ✅ Docker Desktop installed and running
- ✅ Pinecone account ([sign up free](https://app.pinecone.io/))
- ✅ OpenAI API key ([get one here](https://platform.openai.com/))
- ✅ 8GB RAM minimum (16GB recommended)

### Step 1: Configure Environment

Create a `.env` file in the project root:

```bash
# Required for RAG System
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_INDEX=f1-rag-index
OPENAI_API_KEY=your-openai-api-key-here

# Optional - System Configuration
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
OLLAMA_MODEL=llama3.1
REDIS_HOST=redis
API_URL=http://localhost:8000
```

**Getting Your API Keys:**
- **Pinecone**: Login to [Pinecone Console](https://app.pinecone.io/) → Copy API key → Create an index
- **OpenAI**: Visit [OpenAI Platform](https://platform.openai.com/api-keys) → Create new secret key

### Step 2: Start the System

```bash
# Run setup (first time only)
./scripts/setup.sh

# Start all services
./scripts/start.sh
```

**First run**: Takes 5-10 minutes (downloads models and dependencies)  
**Subsequent runs**: 30-60 seconds

### Step 3: Open the Application

Visit: **http://localhost:3000**

You now have a working chat interface! 🎉

---

## System Architecture

```
┌──────────────────────────────────────────────────────┐
│                  User's Browser                      │
└─────────────────────┬────────────────────────────────┘
                      │
                      ↓
┌──────────────────────────────────────────────────────┐
│        Frontend (Next.js) - Port 3000                │
│        • Chat interface                              │
│        • Strategy selection                          │
│        • Streaming responses                         │
└─────────────────────┬────────────────────────────────┘
                      │
                      ↓
┌──────────────────────────────────────────────────────┐
│        Kong API Gateway - Port 8000                  │
│        • Rate limiting (10/sec, 100/min)             │
│        • CORS handling                               │
│        • Request routing                             │
└─────────────────────┬────────────────────────────────┘
                      │
                      ↓
┌──────────────────────────────────────────────────────┐
│        RAG Service (FastAPI) - Port 8001             │
│        • LangChain orchestration                     │
│        • 4 retrieval strategies                      │
│        • Conversation memory                         │
│        • Streaming support                           │
└────────────┬─────────────────────────────────────────┘
             │
    ┌────────┴────────┬─────────────┐
    │                 │             │
    ↓                 ↓             ↓
┌─────────┐      ┌─────────┐   ┌──────────┐
│ OpenAI  │      │Pinecone │   │  Redis   │
│ (LLM &  │      │(Vectors)│   │ (Memory) │
│Embeddings)│    │ (Cloud) │   │  :6379   │
└─────────┘      └─────────┘   └──────────┘

         [Offline Data Loading]
                  │
    ┌─────────────┴───────────────┐
    │  data-loading/ scripts       │
    │  • Load F1 historical data   │
    │  • Process and embed         │
    │  • Upload to Pinecone        │
    └──────────────────────────────┘
```

### File Structure

```
BuildWatch/
│
├── .env                              # Configuration
├── docker-compose.yml                # Service orchestration
│
├── scripts/                          # Automation scripts
│   ├── setup.sh                      # Initial setup
│   ├── start.sh                      # Start all services
│   ├── stop.sh                       # Stop services
│   ├── test-api.sh                   # API testing
│   └── seed-data.sh                  # Sample data
│
├── services/                         # Microservices
│   ├── rag-service/                  # Main RAG backend
│   │   ├── app/main.py               # FastAPI + LangChain
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   │
│   ├── kong/                         # API Gateway
│   │   ├── kong.yml                  # Configuration
│   │   └── Dockerfile
│   │
│   └── frontend/                     # Next.js UI
│       ├── src/app/page.tsx          # Chat interface
│       ├── package.json
│       └── Dockerfile
│
├── data-loading/                     # F1 data ingestion
│   ├── ingest_complete_database.py   # Main loader
│   ├── ingest_csv_data.py            # CSV loader
│   ├── requirements-data.txt
│   └── src/                          # Data processing modules
│
├── data/                             # F1 CSV files
│   └── archive/                      # Historical data
│
└── docs/                             # Documentation
    ├── DEPLOYMENT.md
    └── TROUBLESHOOTING.md
```

---

## Setup Instructions

### Option A: Quick Start (No Data Loading)

Use this to try the system first or if you'll upload your own documents:

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 2. Start system
./scripts/setup.sh
./scripts/start.sh

# 3. Open browser
open http://localhost:3000
```

### Option B: With F1 Data

Load comprehensive Formula 1 data before starting:

```bash
# 1. Configure environment (include OPENAI_API_KEY)
cp .env.example .env
# Edit .env with your API keys

# 2. Load F1 data
cd data-loading
pip install -r requirements-data.txt

# Option 2a: Modern data only (2020-present, faster)
python ingest_complete_database.py --modern-only

# Option 2b: Complete history (1950-present, comprehensive)
python ingest_complete_database.py

# 3. Start RAG system
cd ..
./scripts/start.sh

# 4. Open browser
open http://localhost:3000
```

### Verify Installation

```bash
# Check all services are running
docker ps

# Should see these services:
# - rag-service (port 8001)
# - kong (port 8000)
# - frontend (port 3000)
# - redis (port 6379)

# Test API endpoints
./scripts/test-api.sh
```

---

## Loading F1 Data

The data loading system processes F1 data from multiple sources and stores it in Pinecone for semantic search.

### What Data Is Available

- **Historical Races** (1950-2017): Race results, qualifying, driver standings, constructors
- **Modern Races** (2018-2024): Complete race data with telemetry, lap times, pit stops
- **Driver Profiles**: Career statistics, performance metrics, strengths/weaknesses
- **Circuit Information**: Track characteristics, lap times, historical winners
- **Fantasy Data**: Driver value, points potential, team recommendations

### Data Sources

1. **CSV Files**: Historical data (1950-2017) from Ergast database
2. **FastF1 API**: Modern races with detailed telemetry (2018-2024)
3. **OpenF1 API**: Real-time race data and additional metrics

### Loading Commands

```bash
cd data-loading

# Install dependencies
pip install -r requirements-data.txt

# Complete database (1950-latest)
# Time: ~45-60 minutes | Cost: ~$10-12
python ingest_complete_database.py

# Modern data only (2020-latest) - Good for testing
# Time: ~20-30 minutes | Cost: ~$3-4
python ingest_complete_database.py --modern-only

# Update with latest races only
# Time: ~5-10 minutes | Cost: ~$1-2
python ingest_complete_database.py --update-only

# CSV historical data only
python ingest_csv_data.py

# Fantasy-optimized data
python ingest_fantasy_data.py
```

### What Happens During Loading

1. **Data Collection**: Downloads from APIs or reads CSV files
2. **Processing**: Cleans, structures, and enriches data
3. **Chunking**: Splits into optimal chunks for retrieval
4. **Embedding**: Creates vector embeddings using OpenAI
5. **Upload**: Stores in Pinecone with metadata
6. **Caching**: Saves processed data locally for faster updates

### Output

- `data/f1_unified_complete_database.json` - Complete processed dataset
- `data/f1_dataset_statistics.json` - Dataset statistics
- Pinecone vector database - Searchable embeddings

---

## Using the System

### Chat Interface

**URL**: http://localhost:3000

Features:
- **Text Input**: Type your question and press Enter or click Send
- **Strategy Selection**: Choose how the system searches for information
- **Streaming Toggle**: Enable/disable real-time response streaming
- **Test Strategies**: Compare all 4 strategies side-by-side
- **Clear Chat**: Reset conversation history

### Example Questions

**F1 History:**
```
- Who won the 2023 Monaco Grand Prix?
- What are Lewis Hamilton's career statistics?
- Compare Max Verstappen and Charles Leclerc at Silverstone
- Tell me about the 2021 championship battle
- What happened at Spa in 1998?
```

**Fantasy F1:**
```
- Best driver picks for Silverstone this weekend?
- Which drivers offer the best value?
- Red Bull's performance at Monaco historically?
- Top 5 drivers for wet conditions
```

**Technical Questions:**
```
- What is DRS and how does it work?
- Explain tire strategy in F1
- How do points work in Formula 1?
- What are the differences between soft, medium, and hard tires?
```

### Retrieval Strategies

The system offers 4 different ways to search for information:

#### 1. Similarity Search (⚡⚡⚡ Very Fast)
**How it works**: Direct cosine similarity between your question and stored documents

**Best for**:
- Straightforward questions
- Quick answers
- Specific facts

**Example**: "Who won Monaco 2023?"

#### 2. MMR - Maximal Marginal Relevance (⚡⚡ Medium)
**How it works**: Balances relevance with diversity to avoid redundant results

**Best for**:
- Getting different perspectives
- Exploring multiple aspects
- Comprehensive understanding

**Example**: "Tell me about Lewis Hamilton's career"

#### 3. Multi-Query (⚡ Slower)
**How it works**: Generates multiple query variations, searches for each

**Best for**:
- Complex or ambiguous questions
- Thorough research
- Questions that could be asked multiple ways

**Example**: "Compare Verstappen and Leclerc's driving styles"

#### 4. Compression (⚡ Slower)
**How it works**: Retrieves more documents, then uses LLM to extract relevant parts

**Best for**:
- Long documents
- Extracting specific details
- Reducing noise

**Example**: "Summarize the 2023 F1 season"

### API Endpoints

#### Chat (Non-streaming)
```bash
curl -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Who won Monaco 2023?",
    "retrieval_strategy": "similarity",
    "stream": false,
    "top_k": 5
  }'
```

#### Chat (Streaming)
```bash
curl -N -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Explain DRS",
    "retrieval_strategy": "mmr",
    "stream": true
  }'
```

#### List Strategies
```bash
curl http://localhost:8000/api/retrieval-strategies
```

#### Test All Strategies
```bash
curl -X POST http://localhost:8000/api/test-strategies \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Who won the 2023 championship?"
  }'
```

#### Upload Documents
```bash
# Text documents
curl -X POST http://localhost:8000/api/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "documents": [{
      "content": "Your document text here",
      "metadata": {"source": "custom", "date": "2024-01-01"}
    }]
  }'

# PDF files
curl -X POST http://localhost:8000/api/ingest/pdf \
  -F "file=@/path/to/document.pdf"
```

#### Database Statistics
```bash
curl http://localhost:8000/api/stats
```

#### Health Check
```bash
curl http://localhost:8001/health
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# ==================== Required ====================

# Pinecone Vector Database
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX=f1-rag-index
PINECONE_ENVIRONMENT=us-east-1-aws

# OpenAI API (for embeddings and LLM)
OPENAI_API_KEY=your-openai-api-key

# ==================== Optional ====================

# LLM Provider: "openai" or "ollama"
LLM_PROVIDER=openai

# OpenAI Model (if using OpenAI)
OPENAI_MODEL=gpt-4o-mini

# Ollama Configuration (if using Ollama)
OLLAMA_MODEL=llama3.1
OLLAMA_HOST=http://ollama:11434

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# API Gateway
API_URL=http://localhost:8000

# Web Search
WEB_SEARCH_ENABLED=true
WEB_SEARCH_MAX_RESULTS=5
```

### Using Ollama (Local LLM)

To use Ollama instead of OpenAI:

```bash
# 1. Update .env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1

# 2. Start with Ollama profile
docker compose --profile ollama up -d

# 3. Wait for model download (first time: 5-10 minutes)
docker logs rag-ollama -f
```

**Ollama Models Available:**
- `llama3.1` - Best overall (4GB)
- `llama2` - Older but reliable (3.8GB)
- `mistral` - Fast and accurate (4.1GB)
- `codellama` - Good for code-related questions (3.8GB)

### Port Configuration

Edit `docker-compose.yml` to change ports:

```yaml
services:
  frontend:
    ports:
      - "3000:3000"  # Change first number to use different port
  
  kong:
    ports:
      - "8000:8000"  # Kong Gateway
  
  rag-service:
    ports:
      - "8001:8001"  # RAG Service
```

### Performance Tuning

In `services/rag-service/app/main.py`:

```python
# Number of documents to retrieve
top_k = 5  # Increase for more context, decrease for speed

# Chunk size for documents
chunk_size = 1000  # Larger = more context, smaller = more precise
chunk_overlap = 200  # Overlap between chunks

# LLM temperature
temperature = 0.7  # Lower = more focused, higher = more creative
```

---

## Troubleshooting

### Common Startup Issues

#### Port Already in Use

**Problem**: Error: "Port 3000 is already in use"

**Solution**:
```bash
# Find what's using the port
lsof -i :3000

# Kill the process
kill -9 <PID>

# Or change the port in docker-compose.yml
```

#### Docker Not Running

**Problem**: "Cannot connect to Docker daemon"

**Solution**:
```bash
# Start Docker Desktop (macOS)
open -a Docker

# Verify Docker is running
docker info
```

#### Services Won't Start

**Problem**: Services fail to start or are unhealthy

**Solution**:
```bash
# Check service logs
docker-compose logs

# Check specific service
docker-compose logs rag-service

# Restart all services
docker-compose down
docker-compose up -d
```

### Connection Issues

#### Cannot Connect to Pinecone

**Problem**: "Authentication failed" or "Index not found"

**Solution**:
1. Verify API key in `.env`
2. Check index name matches Pinecone console
3. Verify index dimension is correct (3072 for text-embedding-3-large)
4. Ensure Pinecone environment is correct

```bash
# Test Pinecone connection
python3 << EOF
from pinecone import Pinecone
import os
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
print(pc.list_indexes())
