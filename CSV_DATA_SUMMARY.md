# F1 CSV Data Analysis & Ingestion Summary

## Dataset Overview

Your archive folder contains a comprehensive F1 historical database from **1950-2024** with 14 CSV files:

### Core Data Files

| File | Records | Coverage | Description |
|------|---------|----------|-------------|
| **races.csv** | 1,125 | 1950-2024 | Race calendar, dates, circuits |
| **results.csv** | 26,759 | 1950-2024 | Race finishing positions, points, times |
| **drivers.csv** | 861 | All-time | Driver profiles, nationalities |
| **constructors.csv** | 213 | All-time | Team information |
| **circuits.csv** | 77 | All-time | Circuit locations, coordinates |

### Performance Data Files

| File | Records | Coverage | Description |
|------|---------|----------|-------------|
| **qualifying.csv** | 10,495 | 1950-2024 | Qualifying session results (Q1, Q2, Q3) |
| **lap_times.csv** | 589,082 | 1996-2024 | Individual lap times (very large) |
| **pit_stops.csv** | 11,372 | 2011-2024 | Pit stop durations and timing |
| **sprint_results.csv** | 361 | 2021-2024 | Sprint race results |

### Championship Data Files

| File | Records | Coverage | Description |
|------|---------|----------|-------------|
| **driver_standings.csv** | 34,864 | 1950-2024 | Championship standings after each race |
| **constructor_standings.csv** | 13,392 | 1958-2024 | Constructor championship standings |

### Reference Data Files

| File | Records | Description |
|------|---------|-------------|
| **seasons.csv** | 76 | Season information and URLs |
| **status.csv** | 140 | Finish status codes (Finished, DNF, etc.) |
| **constructor_results.csv** | 12,626 | Team-level race results |

## Data Statistics

### Temporal Coverage
- **Earliest race**: 1950 British Grand Prix
- **Latest race**: 2024 Abu Dhabi Grand Prix
- **Total seasons**: 75 years
- **Total races**: 1,125 race weekends

### 2024 Season Data
- **Races in 2024**: 24 races
- **Complete data**: Race results, qualifying, pit stops, standings
- **Example**: 2024 Bahrain GP
  - Winner: Max Verstappen
  - P2: Sergio Pérez
  - P3: Carlos Sainz

### Data Completeness by Era

| Era | Years | Races | Lap Times | Pit Stops | Sprint | Notes |
|-----|-------|-------|-----------|-----------|--------|-------|
| **Modern** | 2011-2024 | ~350 | ✓ Full | ✓ Full | ✓ 2021+ | Complete dataset |
| **Hybrid** | 2014-2010 | ~100 | ✓ Full | ✗ None | ✗ | Pre-pit stop tracking |
| **Recent** | 1996-2009 | ~250 | ✓ Full | ✗ None | ✗ | Lap times available |
| **Classic** | 1950-1995 | ~425 | ✗ None | ✗ None | ✗ | Results & standings only |

## Data Relationships

The CSV files form a relational database:

```
races.csv (PRIMARY)
  ├─ circuitId → circuits.csv (circuit details)
  │
  ├─ raceId → results.csv (race results)
  │            ├─ driverId → drivers.csv
  │            ├─ constructorId → constructors.csv
  │            └─ statusId → status.csv
  │
  ├─ raceId → qualifying.csv (qualifying results)
  │            └─ driverId → drivers.csv
  │
  ├─ raceId → lap_times.csv (lap times)
  │            └─ driverId → drivers.csv
  │
  ├─ raceId → pit_stops.csv (pit stops)
  │            └─ driverId → drivers.csv
  │
  ├─ raceId → sprint_results.csv (sprint races)
  │            ├─ driverId → drivers.csv
  │            └─ constructorId → constructors.csv
  │
  ├─ raceId → driver_standings.csv
  │            └─ driverId → drivers.csv
  │
  └─ raceId → constructor_standings.csv
               └─ constructorId → constructors.csv
```

## What Gets Ingested into Vector DB

The ingestion script transforms this relational data into **semantic documents** for vector search:

### 1. Race Overview Documents (1 per race)
For each race, creates a rich text document containing:
- Race name, date, circuit, location, country
- Winner and podium finishers
- Weather conditions (when available from other sources)
- Pit stop statistics (average duration)
- Fastest lap information
- Safety car/red flag incidents (when tracked)

**Example**:
```
"Race: Bahrain Grand Prix 2024 | Circuit: Bahrain International Circuit, Bahrain |
Date: 2024-03-02 | Winner: Max Verstappen (Red Bull Racing) |
Podium: Max Verstappen, Sergio Pérez, Carlos Sainz | Average pit stop: 2.4s |
Fastest lap: Charles Leclerc - 1:32.445"
```

### 2. Strategy Documents (1 per driver per race)
For drivers with pit stop data, creates detailed strategy documents:
- Driver name and code
- Number of pit stops
- Stop laps and timing
- Grid position vs finish position
- Points scored
- Circuit and conditions context

**Example**:
```
"Driver: VER | Race: Bahrain Grand Prix 2024 | Circuit: Bahrain International Circuit |
Strategy: 2 stop(s) | Stop 1: Lap 14 | Stop 2: Lap 35 |
Result: P1 (started P1) | Points: 26"
```

### 3. Championship Context
- Driver standings after each race
- Constructor standings progression
- Points and wins tracking

## Ingestion Process

The script performs:

1. **Data Loading**: Reads all CSV files into pandas DataFrames
2. **Data Joining**: Merges related tables (drivers, teams, circuits)
3. **Transformation**: Converts to structured race dictionaries
4. **Document Creation**: Builds semantic text documents
5. **Embedding Generation**: Uses OpenAI text-embedding-3-large (3072 dimensions)
6. **Vector Upload**: Batches to Pinecone with metadata
7. **JSON Export**: Saves processed data for backup

## Recommendations

### For Best Results

**Recommended: Modern Era (2010-2024)**
```bash
python ingest_csv_data.py --start-year 2010
```
- ~300 races with complete data
- All modern regulations and teams
- Pit stop data available (2011+)
- Most relevant for predictions
- Cost: ~$2-3 in OpenAI embeddings

**For Recent Focus (2020-2024)**
```bash
python ingest_csv_data.py --start-year 2020
```
- ~100 races, current regulations
- All modern cars and drivers
- Sprint race data included
- Cost: ~$1 in embeddings

**For Single Season Testing (2024)**
```bash
python ingest_csv_data.py --year 2024
```
- 24 races, current season only
- Great for testing/validation
- Cost: <$0.50 in embeddings

**For Full Historical Context (1950-2024)**
```bash
python ingest_csv_data.py --all-history
```
- 1,125 races, complete F1 history
- Useful for historical comparisons
- ~30-60 minutes processing time
- Cost: ~$8-12 in embeddings

## Output & Results

After ingestion, you'll have:

### 1. Vector Database (Pinecone)
- Searchable embeddings for all races
- Rich metadata for filtering
- Semantic search capabilities
- Ready for RAG applications

**Expected vector counts**:
- 2024 only: ~400-500 vectors
- 2020-2024: ~2,000-2,500 vectors
- 2010-2024: ~4,500-5,500 vectors
- 1950-2024: ~15,000-20,000 vectors

### 2. Processed JSON Files
Location: `./data/f1_csv_processed_[start]_[end].json`

Contains structured data for:
- Model training
- Analysis
- Debugging
- Re-ingestion without reprocessing

### 3. Log Files
Location: `./logs/csv_ingestion.log`

Detailed logs of:
- Data loading progress
- Embedding generation
- Vector uploads
- Any errors or warnings

## Usage Examples

### Quick Start (Recommended)
```bash
# Modern era ingestion
python ingest_csv_data.py
```

### Specific Use Cases
```bash
# Just 2024 for testing
python ingest_csv_data.py --year 2024

# Last 5 years
python ingest_csv_data.py --start-year 2020

# Include lap time data (memory intensive)
python ingest_csv_data.py --start-year 2022 --include-lap-times

# Full history
python ingest_csv_data.py --all-history
```

### Python API
```python
from src.csv_data_ingestion import F1CSVIngestion

# Initialize
ingestion = F1CSVIngestion(csv_data_dir='./data/archive')

# Ingest specific years
ingestion.ingest_all(
    start_year=2020,
    end_year=2024,
    include_pit_stops=True,
    batch_size=50
)

# Check results
stats = ingestion.vector_db.get_stats()
print(f"Vectors indexed: {stats['total_vectors']}")
```

## Next Steps After Ingestion

1. **Verify Ingestion**
   ```python
   from src.vector_db import F1VectorDB

   vdb = F1VectorDB()
   stats = vdb.get_stats()
   print(f"Total vectors: {stats['total_vectors']}")
   ```

2. **Test Semantic Search**
   ```python
   results = vdb.search_similar_races(
       query="Monaco wet conditions safety car",
       top_k=5
   )
   ```

3. **Build RAG System**
   - Use vectors for context retrieval
   - Feed to LLM for race predictions
   - Strategy recommendations

4. **Train ML Models**
   - Use processed JSON for features
   - Historical patterns for predictions
   - Strategy optimization

## Cost Estimates

| Scope | Races | Vectors | OpenAI Cost | Time |
|-------|-------|---------|-------------|------|
| 2024 only | 24 | ~500 | $0.30 | 2-3 min |
| 2020-2024 | ~100 | ~2,000 | $1.00 | 5-10 min |
| 2010-2024 | ~300 | ~5,000 | $2.50 | 15-20 min |
| Full history | 1,125 | ~18,000 | $10.00 | 45-60 min |

*Pinecone free tier includes 100k vectors (sufficient for all data)*

## Files Created

1. **[src/csv_data_ingestion.py](src/csv_data_ingestion.py)** - Main ingestion module
2. **[ingest_csv_data.py](ingest_csv_data.py)** - Quick start script
3. **[CSV_INGESTION_GUIDE.md](CSV_INGESTION_GUIDE.md)** - Detailed guide
4. **[CSV_DATA_SUMMARY.md](CSV_DATA_SUMMARY.md)** - This file

## Integration

This CSV ingestion complements your existing [knowledge_base_builder.py](src/knowledge_base_builder.py):

- **CSV ingestion**: Historical data (1950-2024) from archive
- **API-based builder**: Live data with telemetry (2018+) from FastF1

Use both together for complete coverage:
- Historical context: 1950-2017 (CSV)
- Modern detailed data: 2018-2024 (API with telemetry)
