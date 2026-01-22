# F1 CSV Data Ingestion Guide

This guide explains how to ingest the historical F1 CSV data from the archive folder into your vector database.

## Data Overview

The archive folder contains comprehensive F1 data from **1950-2024**:

| File | Rows | Description |
|------|------|-------------|
| `races.csv` | 1,126 | Race calendar and basic information |
| `results.csv` | 26,760 | Race results and finishing positions |
| `drivers.csv` | 862 | Driver information and details |
| `constructors.csv` | 213 | Team/constructor information |
| `circuits.csv` | 78 | Circuit details and locations |
| `qualifying.csv` | 10,495 | Qualifying session results |
| `lap_times.csv` | 589,082 | Individual lap times (large dataset) |
| `pit_stops.csv` | 11,372 | Pit stop data |
| `sprint_results.csv` | 361 | Sprint race results |
| `driver_standings.csv` | 34,864 | Championship standings (drivers) |
| `constructor_standings.csv` | 13,392 | Championship standings (constructors) |
| `status.csv` | 140 | Race finish status codes |
| `seasons.csv` | 76 | Season information |
| `constructor_results.csv` | 12,626 | Team race results |

## Data Relationships

The CSV data is relational and interconnected:

```
races.csv (race_id, circuit_id, year, round)
  ├── results.csv (race_id → driver_id, constructor_id)
  ├── qualifying.csv (race_id → driver_id)
  ├── pit_stops.csv (race_id → driver_id)
  ├── lap_times.csv (race_id → driver_id)
  ├── sprint_results.csv (race_id → driver_id, constructor_id)
  └── standings.csv (race_id → driver_id/constructor_id)

drivers.csv (driver_id, name, nationality)
constructors.csv (constructor_id, name, nationality)
circuits.csv (circuit_id, name, location, country)
```

## Quick Start

### 1. Ensure Environment Variables are Set

```bash
# .env file should contain:
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

### 2. Basic Ingestion (2010-2024, recommended)

Ingest modern F1 data from 2010 onwards:

```bash
python src/csv_data_ingestion.py --start-year 2010
```

This will:
- Load races from 2010 to latest (2024)
- Include race results, qualifying, pit stops
- Generate embeddings for ~300 races
- Ingest into your vector database
- Save processed data to `data/f1_csv_processed_2010_latest.json`

### 3. Full Historical Ingestion (1950-2024)

Ingest all available historical data:

```bash
python src/csv_data_ingestion.py --start-year 1950
```

**Note:** This will process 1,126 races and generate thousands of vectors.

### 4. Specific Year Range

```bash
# Just 2020-2023
python src/csv_data_ingestion.py --start-year 2020 --end-year 2023

# Only 2024 season
python src/csv_data_ingestion.py --start-year 2024 --end-year 2024
```

## Advanced Options

### Include Lap Times (Memory Intensive)

```bash
python src/csv_data_ingestion.py --start-year 2020 --include-lap-times
```

**Warning:** Lap times add 589k+ records. Only use for recent years or you may run out of memory.

### Adjust Batch Size

Control how many vectors are uploaded at once:

```bash
python src/csv_data_ingestion.py --start-year 2010 --batch-size 100
```

Larger batches = faster but more memory usage.

### Custom CSV Directory

If your CSV files are in a different location:

```bash
python src/csv_data_ingestion.py --csv-dir /path/to/csv/files
```

## What Gets Ingested?

For each race, the script creates:

1. **Race Overview Document** - Embedded with:
   - Race name, date, circuit, country
   - Race winner and podium
   - Weather conditions (if available)
   - Key statistics (fastest lap, pit stop averages)
   - Safety car/red flag incidents

2. **Driver Strategy Documents** - One per driver with:
   - Pit stop strategy (number of stops, stop laps)
   - Grid position vs finish position
   - Race result and points
   - Circuit and conditions context

These are converted to embeddings and stored in Pinecone with rich metadata for semantic search.

## Data Processing Details

The ingestion script:

1. **Loads CSV files** into pandas DataFrames
2. **Joins related data** (drivers, constructors, circuits)
3. **Transforms into structured format** matching the vector DB schema
4. **Generates embeddings** using OpenAI text-embedding-3-large
5. **Batches uploads** to Pinecone with metadata
6. **Saves processed JSON** for backup and debugging

## Recommended Approach

For best results, we recommend ingesting in stages:

```bash
# Stage 1: Modern era (2014-2024) - Current regulations
python src/csv_data_ingestion.py --start-year 2014

# Stage 2: Add turbo-hybrid era (2010-2013)
python src/csv_data_ingestion.py --start-year 2010 --end-year 2013

# Stage 3: Historical data if needed
python src/csv_data_ingestion.py --start-year 1950 --end-year 2009
```

This keeps recent, relevant data prioritized while allowing historical context.

## Monitoring Progress

The script provides detailed progress information:

```
Loading CSV files from ./data/archive...
✓ Loaded 1126 races from CSV files
Processing 300 races (years: 2010-all)...
Loading races: 100%|████████| 300/300 [00:45<00:00, 6.67it/s]
✓ Loaded 300 races successfully

[2/2] Ingesting into vector database...
Processing races: 100%|████████| 300/300 [15:23<00:00, 3.08s/it]
Generating embeddings: 100%|████████| 42/42 [02:15<00:00, 3.23s/it]
✓ Ingestion complete

📊 SUMMARY:
   • Races processed: 300
   • Year range: 2010 - latest
   • Vector database: 4,873 vectors indexed
```

## Output Files

After ingestion, you'll have:

1. **Vector Database**: All embeddings stored in Pinecone
2. **Processed JSON**: `data/f1_csv_processed_YEAR_YEAR.json`
   - Complete structured data
   - Useful for debugging and analysis
   - Can be re-ingested without re-processing CSVs
3. **Logs**: `logs/csv_ingestion.log`

## Troubleshooting

### "Data directory not found"
Make sure CSV files are in `./data/archive/` or specify custom path with `--csv-dir`

### "OPENAI_API_KEY not set"
Add your OpenAI API key to `.env` file

### "Memory error" during ingestion
- Reduce year range (process fewer races)
- Don't use `--include-lap-times` flag
- Reduce `--batch-size`

### Vector count seems low
The script creates:
- 1 race overview per race
- 1-20 driver strategies per race (depending on finishers)

So 300 races ≈ 300 + (300 × 15 avg drivers) = ~4,800 vectors

## Integration with Existing System

This CSV ingestion is compatible with the existing [knowledge_base_builder.py](src/knowledge_base_builder.py):

- **CSV ingestion**: Historical data (1950-2024) from archive
- **API-based collection**: Live data from FastF1/Ergast APIs (2017-present)

You can use both systems together:
1. Ingest historical CSV data (1950-2016)
2. Use knowledge_base_builder for recent seasons (2017+)

## Next Steps

After ingestion:

1. **Verify data**: Check vector count with `vector_db.get_stats()`
2. **Test queries**: Run semantic searches for races/strategies
3. **Build RAG system**: Use vectors for context retrieval
4. **Train models**: Use processed JSON for ML model training

## Example Usage in Python

```python
from src.csv_data_ingestion import F1CSVIngestion

# Initialize
ingestion = F1CSVIngestion(csv_data_dir='./data/archive')

# Ingest 2020-2024 seasons
ingestion.ingest_all(
    start_year=2020,
    end_year=2024,
    include_pit_stops=True,
    batch_size=50
)

# Check results
stats = ingestion.vector_db.get_stats()
print(f"Total vectors: {stats['total_vectors']}")
```

## Cost Estimates

Approximate costs for OpenAI embeddings (text-embedding-3-large):

- **2010-2024** (~300 races): ~$2-4
- **Full history** (1,126 races): ~$8-12

Pinecone free tier includes 100k vectors, which is sufficient for all F1 historical data.
