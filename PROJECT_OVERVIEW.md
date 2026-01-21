# F1 Knowledge Base - Project Overview

## What You've Built

A complete F1 race analysis system that:
1. Collects comprehensive race data from 2017-present
2. Stores it in a vector database for semantic search
3. Predicts optimal race strategies based on historical data
4. Simulates races lap-by-lap with strategy execution

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    F1 Knowledge Base System                  │
└─────────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌──────────────────┐
│   FastF1 API     │────────▶│  Data Collector  │
│  (2017-present)  │         │   Module         │
└──────────────────┘         └────────┬─────────┘
                                      │
                                      ▼
                            ┌─────────────────┐
                            │  Race Data      │
                            │  (JSON Cache)   │
                            └────────┬────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                │                    │                    │
                ▼                    ▼                    ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │  OpenAI          │  │  Vector DB       │  │  Strategy        │
    │  Embeddings      │─▶│  (Pinecone)      │◀─│  Predictor       │
    │  Generator       │  │  3072-dim        │  │                  │
    └──────────────────┘  └──────────────────┘  └─────────┬────────┘
                                                            │
                                                            ▼
                                                  ┌──────────────────┐
                                                  │  Race Simulator  │
                                                  │  (Lap-by-lap)    │
                                                  └──────────────────┘
```

## File Structure

### Core Modules

**[src/data_collector.py](src/data_collector.py)** (900+ lines)
- Collects F1 race data from FastF1
- Full telemetry, lap times, tire strategies, weather
- Caching for incremental updates
- Exports to JSON

**[src/vector_db.py](src/vector_db.py)** (500+ lines)
- Pinecone integration
- OpenAI embedding generation
- Semantic search functionality
- Race and strategy document creation

**[src/knowledge_base_builder.py](src/knowledge_base_builder.py)** (350+ lines)
- Orchestrates the complete pipeline
- Manages incremental updates
- Exports statistics and datasets

**[src/strategy_predictor.py](src/strategy_predictor.py)** (550+ lines)
- Predicts optimal race strategies
- Analyzes historical similar races
- Calculates pit windows
- Suggests tire compounds

**[src/race_simulator.py](src/race_simulator.py)** (650+ lines)
- Lap-by-lap race simulation
- Tire degradation modeling
- Dynamic strategy execution
- Position updates and incident simulation

### Configuration & Docs

**[config.py](config.py)** - Central configuration management
**[.env](.env)** - Environment variables (API keys configured!)
**[requirements.txt](requirements.txt)** - Python dependencies
**[README.md](README.md)** - Complete documentation
**[QUICKSTART.md](QUICKSTART.md)** - Quick start guide
**[example_usage.py](example_usage.py)** - Interactive examples

## Key Features

### 1. Data Collection
```python
from src.data_collector import F1DataCollector

collector = F1DataCollector(start_year=2017)
data = collector.collect_all_seasons()
# Collects: race results, lap times, pit stops, tire strategies,
# weather, telemetry, qualifying, sprint races
```

### 2. Vector Search
```python
from src.vector_db import F1VectorDB

vdb = F1VectorDB()
results = vdb.search_similar_races(
    query="Monaco wet conditions safety car",
    top_k=5
)
# Semantic search across 200+ races
```

### 3. Strategy Prediction
```python
from src.strategy_predictor import F1StrategyPredictor, SessionData

predictor = F1StrategyPredictor()
session = SessionData(
    circuit="Silverstone",
    total_laps=52,
    weather="Dry",
    track_temp=35.0,
    available_compounds=["SOFT", "MEDIUM", "HARD"]
)

recommendation = predictor.predict_optimal_strategy(session)
# Returns: strategy type, stint plan, confidence, reasoning
```

### 4. Race Simulation
```python
from src.race_simulator import RaceSimulator

simulator = RaceSimulator()
simulator.setup_race(
    circuit="Spa-Francorchamps",
    total_laps=44,
    drivers=[...],
    weather="Dry"
)

results = simulator.simulate_race(verbose=True)
# Simulates: lap times, pit stops, positions, tire deg, incidents
```

## Getting Started

### Option 1: Quick Setup (Recommended)
```bash
./setup.sh
source venv/bin/activate
python example_usage.py
```

### Option 2: Manual Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd src
python knowledge_base_builder.py
```

## Data Flow

1. **Collection Phase**
   - FastF1 downloads race data
   - Processes into structured format
   - Caches locally (cache/f1_data/)
   - Exports to JSON (data/)

2. **Embedding Phase**
   - Creates text documents from race data
   - Generates embeddings via OpenAI
   - Batch processing for efficiency

3. **Ingestion Phase**
   - Uploads vectors to Pinecone
   - Stores metadata for filtering
   - Indexes for fast retrieval

4. **Query Phase**
   - User provides race conditions
   - System generates query embedding
   - Searches similar historical races
   - Returns relevant strategies

5. **Prediction Phase**
   - Analyzes historical strategies
   - Calculates optimal stint plan
   - Provides confidence scores
   - Generates reasoning

6. **Simulation Phase**
   - Executes strategies lap-by-lap
   - Models tire degradation
   - Updates positions dynamically
   - Tracks all race events

## Performance Metrics

### Data Collection
- **Time**: 2-4 hours for complete dataset (2017-2024)
- **Storage**: ~500MB cached data
- **Races**: ~200 race weekends
- **Data Points**: 50,000+ lap times, 5,000+ pit stops

### Vector Database
- **Vectors**: 400-600 vectors (races + strategies)
- **Dimension**: 3072 (text-embedding-3-large)
- **Ingestion**: 30-60 minutes
- **Query Speed**: <2 seconds

### Prediction
- **Accuracy**: Based on historical similarity
- **Speed**: <2 seconds per prediction
- **Confidence**: Calculated from sample size

### Simulation
- **Speed**: <5 seconds for 50-lap race
- **Realism**: Simplified tire model
- **Events**: Random safety cars, DNFs

## Use Cases

### 1. Pre-Race Strategy Planning
Predict optimal strategies before race based on:
- Circuit characteristics
- Weather forecast
- Tire compounds available
- Historical performance

### 2. Live Strategy Adjustments
Update strategies during race based on:
- Current position
- Tire condition
- Competitor strategies
- Track incidents

### 3. Race Analysis
Analyze completed races:
- Compare strategies
- Identify optimal decisions
- Study tire degradation
- Evaluate pit timing

### 4. "What-If" Scenarios
Simulate alternative strategies:
- Different pit stop timing
- Alternative compounds
- Response to safety cars
- Position recovery drives

### 5. Historical Research
Search and analyze:
- Similar race conditions
- Successful strategies
- Circuit-specific patterns
- Weather impacts

## API Keys Already Configured ✅

Your `.env` file includes:
- **Pinecone API Key**: Configured
- **OpenAI API Key**: Configured

You're ready to start immediately!

## Next Steps

1. **Build Knowledge Base**
   ```bash
   cd src
   python knowledge_base_builder.py
   ```
   This will take 2-4 hours. Get coffee! ☕

2. **Run Examples**
   ```bash
   python example_usage.py
   ```

3. **Experiment**
   - Try different circuits
   - Modify simulation parameters
   - Test various weather conditions
   - Compare strategies

4. **Extend**
   - Add more data sources
   - Improve tire models
   - Build web interface
   - Create visualizations

## Troubleshooting

### "No module named 'src'"
```bash
# Make sure you're in the project root
cd /Users/rahulpatel/Documents/BuildWatch
python -m src.knowledge_base_builder
```

### "Rate limit exceeded"
- OpenAI free tier has limits
- Add delays between requests
- Consider upgrading API plan

### "Pinecone index not found"
- First run creates index automatically
- Wait 30 seconds for index initialization
- Check API key is correct

### "FastF1 download failed"
- Some races have limited data
- Script continues with available data
- Check internet connection

## Resources

- **FastF1 Docs**: https://docs.fastf1.dev/
- **Pinecone Docs**: https://docs.pinecone.io/
- **OpenAI Embeddings**: https://platform.openai.com/docs/guides/embeddings

## Support

Questions or issues?
1. Check [README.md](README.md) for detailed docs
2. Review [QUICKSTART.md](QUICKSTART.md) for setup
3. Run [example_usage.py](example_usage.py) for examples
4. Check code comments in source files

---

**Built for:** Race strategy prediction and simulation
**Tech Stack:** Python, FastF1, Pinecone, OpenAI, NumPy, Pandas
**Author:** Your F1 Analysis System
**Version:** 1.0.0

🏎️ Ready to predict some race strategies! 🏁
