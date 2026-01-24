# F1 Data Loading Scripts

This directory contains scripts to load F1 historical data into your Pinecone vector database.

## Overview

The data loading system combines multiple sources to create a comprehensive F1 database:

- **CSV files** (1950-2017): Historical race data
- **FastF1 API** (2018-2024): Modern races with telemetry
- **Ergast API**: Fallback data source

## Prerequisites

1. **Environment Variables** (in parent `.env`):
   ```bash
   OPENAI_API_KEY=your-openai-key
   PINECONE_API_KEY=your-pinecone-key
   PINECONE_INDEX=your-index-name
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements-data.txt
   ```

3. **CSV Data Files** (optional):
   Place your F1 CSV files in `../data/` directory

## Usage

### Complete Database (1950-latest)

Load all historical and modern data:

```bash
python ingest_complete_database.py
```

**Time**: ~45-60 minutes
**Cost**: ~$10-12 (OpenAI embeddings)

### Modern Data Only (2020-latest)

Quick start with recent data:

```bash
python ingest_complete_database.py --modern-only
```

**Time**: ~20-30 minutes
**Cost**: ~$3-4

### Update with Latest Races

Add only the newest races from current season:

```bash
python ingest_complete_database.py --update-only
```

**Time**: ~5-10 minutes
**Cost**: ~$1-2

### CSV Data Only

Load just the historical CSV files:

```bash
python ingest_csv_data.py
```

### Fantasy Data

Load F1 fantasy-optimized data:

```bash
python ingest_fantasy_data.py
```

## Data Sources

### src/unified_data_ingestion.py
Main orchestrator that combines all data sources.

### src/csv_data_ingestion.py
Processes historical CSV files (races, drivers, constructors, etc.)

### src/fantasy_data_ingestion.py
Processes fantasy-optimized data with driver recommendations.

### src/data_collector.py
Collects data from FastF1 API with telemetry.

### src/openf1_collector.py
Collects data from OpenF1 API.

### src/vector_db.py
Handles Pinecone vector database operations.

### src/knowledge_base_builder.py
Builds knowledge base from collected data.

## What Gets Loaded

After running the complete ingestion, your Pinecone database will contain:

- **Historical Races** (1950-2017): All race results, qualifying, drivers, teams
- **Modern Races** (2018-latest): Race data + telemetry + lap times
- **Driver Profiles**: Performance stats, strengths, weaknesses
- **Circuit Data**: Track characteristics, average lap times
- **Fantasy Metrics**: Driver value, points potential, recommendations
- **Team Performance**: Constructor standings and performance

## Output

Data is saved to:
- `../data/f1_unified_complete_database.json` - Complete dataset
- Pinecone vector database - Searchable vectors for RAG

## Integration with RAG System

Once data is loaded into Pinecone, the RAG microservices can query it:

```bash
# Start RAG system (from parent directory)
cd ..
./scripts/start.sh

# The RAG service will automatically use your Pinecone index
# configured in .env
```

## Options

### --modern-only
Only load data from 2020-present (faster for testing)

### --no-fantasy
Skip fantasy metrics calculation (faster ingestion)

### --update-only
Only fetch and load the latest races from current season

### --force-redownload
Re-download API data even if cached

## Costs

Using OpenAI embeddings (text-embedding-3-small):

- **Complete DB**: ~25,000 chunks × $0.00002/1K tokens = ~$10-12
- **Modern only**: ~8,000 chunks × $0.00002/1K tokens = ~$3-4
- **Update**: ~500 chunks × $0.00002/1K tokens = ~$1-2

## Troubleshooting

### "OPENAI_API_KEY not set"
Add your OpenAI API key to parent `.env` file

### "PINECONE_API_KEY not set"
Add your Pinecone API key to parent `.env` file

### "No CSV files found"
Either:
- Add CSV files to `../data/` directory
- Use `--modern-only` to skip CSV loading

### "Rate limit exceeded"
FastF1 API has rate limits. The script includes automatic retries and delays.

### Import errors
Install dependencies: `pip install -r requirements-data.txt`

## Architecture

```
data-loading/
├── ingest_complete_database.py  ← Main entry point
├── ingest_csv_data.py           ← CSV-only ingestion
├── ingest_fantasy_data.py       ← Fantasy data
├── requirements-data.txt        ← Python dependencies
└── src/
    ├── unified_data_ingestion.py    ← Orchestrator
    ├── csv_data_ingestion.py        ← CSV processing
    ├── fantasy_data_ingestion.py    ← Fantasy processing
    ├── data_collector.py            ← FastF1 API
    ├── openf1_collector.py          ← OpenF1 API
    ├── vector_db.py                 ← Pinecone operations
    └── knowledge_base_builder.py    ← Knowledge base
```

## Next Steps

After loading data:

1. Verify data in Pinecone console
2. Start RAG microservices: `cd .. && ./scripts/start.sh`
3. Test queries in the chat interface at http://localhost:3000
4. Ask questions like:
   - "Who won the 2023 Monaco Grand Prix?"
   - "What are Lewis Hamilton's strengths?"
   - "Best driver picks for Silverstone?"

---

For RAG system documentation, see the parent directory README.
