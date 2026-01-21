# F1 Race Knowledge Base & Strategy Predictor

A comprehensive F1 race analysis system that builds a knowledge base from historical race data (2017-present) and uses vector similarity search to predict optimal race strategies and simulate race outcomes.

## Features

### 📊 Data Collection
- Collects comprehensive F1 race data from 2017 onwards using FastF1
- Full telemetry: lap times, tire strategies, pit stops, weather conditions
- Track status: safety cars, red flags, incidents
- Driver and team performance data
- Qualifying and sprint race results

### 🧠 Vector Knowledge Base
- Powered by Pinecone vector database
- OpenAI text-embedding-3-large embeddings (3072 dimensions)
- Semantic search across historical races
- Retrieval-Augmented Generation (RAG) ready

### 🏎️ Strategy Prediction
- Predicts optimal race strategies based on historical data
- Analyzes similar races using vector similarity search
- Recommends tire compound sequences
- Calculates pit stop windows
- Real-time strategy updates during race progression

### 🎮 Race Simulation
- Lap-by-lap race simulation
- Tire degradation modeling
- Dynamic pit stop decisions
- Position updates based on race time
- Random events (safety cars, DNFs)
- Strategy updates mid-race

## Project Structure

```
BuildWatch/
├── src/
│   ├── data_collector.py          # F1 data collection module
│   ├── vector_db.py                # Vector database integration
│   ├── knowledge_base_builder.py   # Pipeline orchestrator
│   ├── strategy_predictor.py       # Strategy prediction engine
│   └── race_simulator.py           # Race simulation engine
├── cache/                          # Cached data
│   ├── f1_data/                   # Processed race data
│   └── fastf1/                    # FastF1 cache
├── data/                          # Exported datasets
├── logs/                          # Application logs
├── requirements.txt
├── .env.example
└── README.md
```

## Installation

### Prerequisites
- Python 3.9+
- Pinecone account (free tier available)
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd BuildWatch
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:
- `PINECONE_API_KEY`: Your Pinecone API key
- `OPENAI_API_KEY`: Your OpenAI API key
- `START_YEAR`: Starting year for data collection (default: 2017)

## Usage

### 1. Build Knowledge Base

Collect F1 data and ingest into vector database:

```bash
cd src
python knowledge_base_builder.py
```

This will:
1. Collect all F1 race data from 2017-present using FastF1
2. Process and structure the data
3. Generate embeddings using OpenAI
4. Ingest into Pinecone vector database

**Note:** Initial build can take 2-4 hours depending on your internet connection and API rate limits.

### 2. Predict Race Strategy

```python
from src.strategy_predictor import F1StrategyPredictor, SessionData

# Initialize predictor
predictor = F1StrategyPredictor()

# Define current session
session = SessionData(
    circuit="Monaco",
    session_type="Race",
    lap_number=1,
    total_laps=78,
    air_temp=22.0,
    track_temp=38.0,
    weather="Dry",
    available_compounds=["SOFT", "MEDIUM", "HARD"]
)

# Get strategy recommendation
recommendation = predictor.predict_optimal_strategy(session)

print(f"Recommended: {recommendation.strategy_type}")
print(f"Confidence: {recommendation.confidence:.1%}")
print(f"\nStint Plan:")
for stint in recommendation.stints:
    print(f"  Stint {stint['stint_number']}: {stint['compound']} "
          f"(Laps {stint['start_lap']}-{stint['end_lap']})")
```

### 3. Simulate Race

```python
from src.race_simulator import RaceSimulator

# Initialize simulator
simulator = RaceSimulator()

# Define driver grid
drivers = [
    {'number': '1', 'name': 'Max Verstappen', 'team': 'Red Bull', 'base_lap_time': 78.0},
    {'number': '44', 'name': 'Lewis Hamilton', 'team': 'Mercedes', 'base_lap_time': 78.2},
    {'number': '16', 'name': 'Charles Leclerc', 'team': 'Ferrari', 'base_lap_time': 78.3},
]

# Setup race
simulator.setup_race(
    circuit="Silverstone",
    total_laps=52,
    drivers=drivers,
    weather="Dry",
    track_temp=32.0
)

# Run simulation
results = simulator.simulate_race(verbose=True)

# View results
for classification in results['classifications']:
    print(f"P{classification['position']}: {classification['driver']} - "
          f"{classification['stops']} stops")
```

### 4. Search Similar Races

```python
from src.vector_db import F1VectorDB

# Initialize vector database
vdb = F1VectorDB()

# Search for similar races
results = vdb.search_similar_races(
    query="Monaco street circuit wet conditions multiple safety cars",
    top_k=5
)

for result in results:
    print(f"\nSimilarity: {result['score']:.3f}")
    print(f"Race: {result['text'][:200]}...")
```

### 5. Update with Latest Races

Incrementally update the knowledge base with new races:

```python
from src.knowledge_base_builder import F1KnowledgeBaseBuilder

builder = F1KnowledgeBaseBuilder()
builder.update_with_latest_races()
```

## API Reference

### Data Collector

**F1DataCollector**
- `collect_all_seasons(years)`: Collect data for multiple seasons
- `collect_season(year)`: Collect single season
- `collect_race_weekend(year, round)`: Collect single race
- `export_to_json(data, file)`: Export collected data

### Vector Database

**F1VectorDB**
- `ingest_race_data(races)`: Ingest race data into vector DB
- `search_similar_races(query, top_k)`: Search for similar races
- `search_similar_strategies(query, circuit, top_k)`: Search strategies
- `get_race_context(conditions, top_k)`: Get relevant historical context
- `get_stats()`: Get database statistics

### Strategy Predictor

**F1StrategyPredictor**
- `predict_optimal_strategy(session_data, position)`: Predict strategy
- `predict_pit_window(session_data, compound, stint_start)`: Calculate pit window
- `suggest_next_compound(session_data, remaining_laps, used)`: Recommend compound
- `get_live_strategy_update(session_data, original_plan, lap)`: Live updates

### Race Simulator

**RaceSimulator**
- `setup_race(circuit, laps, drivers, weather, temps)`: Configure race
- `simulate_race(verbose)`: Run complete simulation
- `simulate_lap(verbose)`: Simulate single lap
- `get_live_standings()`: Get current standings
- `update_strategy_mid_race(driver_name)`: Update strategy during race
- `export_lap_chart()`: Export position data

## Data Sources

- **FastF1**: Primary source for detailed telemetry and timing data
- **Ergast API**: Fallback for basic race results (if needed)

## Performance

- **Data Collection**: ~2-4 hours for complete dataset (2017-present)
- **Vector Ingestion**: ~30-60 minutes for ~200 races
- **Strategy Prediction**: <2 seconds per query
- **Race Simulation**: <5 seconds for 50-70 lap race

## Limitations

- FastF1 data availability varies by season (newer = more detailed)
- Telemetry data sampled to reduce storage requirements
- Simulation uses simplified tire degradation model
- Weather data may be incomplete for older races

## Future Enhancements

- [ ] Live timing integration for real-time predictions
- [ ] Advanced tire degradation models (ML-based)
- [ ] Fuel load impact modeling
- [ ] DRS effect simulation
- [ ] Team radio context analysis
- [ ] LLM-powered natural language strategy explanations
- [ ] Web dashboard for visualizations

## Contributing

Contributions welcome! Areas of interest:
- Improved tire degradation models
- Additional data sources
- Better simulation accuracy
- Visualization tools

## License

MIT License

## Acknowledgments

- FastF1 for comprehensive F1 data access
- Pinecone for vector database infrastructure
- OpenAI for embedding models

## Support

For issues or questions:
1. Check existing documentation
2. Search closed issues
3. Open new issue with details

---

Built with ❤️ for F1 strategy analysis
