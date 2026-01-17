# 🏗️ F1 Race Strategy Analyzer - System Architecture

Detailed architecture documentation for the F1 Race Strategy Analyzer system.

## System Overview

The F1 Race Strategy Analyzer is an AI-powered system that combines:
- **Machine Learning** for race predictions
- **Vector Databases** for semantic search over historical data
- **Large Language Models** for natural language strategy generation
- **Event-Driven Architecture** for automated data updates

## High-Level Architecture

```
                                    ┌──────────────────────────────────┐
                                    │         External APIs            │
                                    │  (Ergast, FastF1, OpenF1)        │
                                    └──────────────┬───────────────────┘
                                                   │
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              DATA INGESTION LAYER                                 │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────────────┐ │
│  │  data_collection   │  │  EventBridge       │  │  Lambda: Ingestion         │ │
│  │  - ErgastAPI       │──│  - Race End Trigger│──│  - Scheduled Refresh       │ │
│  │  - FastF1Collector │  │  - Weekly Update   │  │  - On-Demand Collection    │ │
│  │  - OpenF1API       │  └────────────────────┘  └────────────────────────────┘ │
│  └────────────────────┘                                                          │
└──────────────────────────────────────────────────────────────────────────────────┘
                                                   │
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              DATA STORAGE LAYER                                   │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────────────┐ │
│  │      S3 Bucket     │  │     DynamoDB       │  │    Local File System       │ │
│  │  - Raw race data   │  │  - Race metadata   │  │  - ./data/raw              │ │
│  │  - Model artifacts │  │  - Driver stats    │  │  - ./data/processed        │ │
│  │  - Telemetry logs  │  │  - Circuit info    │  │  - ./models                │ │
│  └────────────────────┘  └────────────────────┘  └────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────┘
                                                   │
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           FEATURE ENGINEERING LAYER                               │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                        feature_engineering.py                               │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │ │
│  │  │ CircuitEncoder│  │ TireEncoder  │  │WeatherEncoder│  │DriverEncoder │   │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │ │
│  │                              │                                              │ │
│  │                              ▼                                              │ │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │ │
│  │  │              F1FeatureEngineer                                      │   │ │
│  │  │  - prepare_race_sequence()    - prepare_training_data()            │   │ │
│  │  │  - prepare_pit_stop_features() - normalize_features()              │   │ │
│  │  └────────────────────────────────────────────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────┘
                                                   │
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              AI/ML PROCESSING LAYER                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                         tensorflow_models.py                                 │ │
│  │                                                                              │ │
│  │  ┌────────────────────┐  ┌────────────────────┐  ┌─────────────────────┐   │ │
│  │  │ RaceOutcomePredictor│  │  PitStopOptimizer  │  │TireDegradationModel │   │ │
│  │  │   (LSTM Network)    │  │  (Deep Q-Network)  │  │   (Regression)      │   │ │
│  │  │                     │  │                     │  │                     │   │ │
│  │  │ Input: 10x30 seq    │  │ Input: 15-dim state│  │ Input: 8 features   │   │ │
│  │  │ Output: Position    │  │ Output: 4 actions  │  │ Output: Lap time    │   │ │
│  │  └────────────────────┘  └────────────────────┘  └─────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                         vector_database.py                                   │ │
│  │  ┌────────────────────┐  ┌────────────────────────────────────────────┐    │ │
│  │  │ EmbeddingGenerator │  │         F1VectorDatabase                    │    │ │
│  │  │ - Hash-based       │  │  - Pinecone integration                    │    │ │
│  │  │ - Semantic vectors │  │  - Local fallback storage                  │    │ │
│  │  └────────────────────┘  │  - Similarity search                       │    │ │
│  │                          │  - Strategy indexing                        │    │ │
│  │                          └────────────────────────────────────────────┘    │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                      llm_strategy_generator.py                               │ │
│  │  ┌────────────────────────────────────────────────────────────────────┐    │ │
│  │  │              F1StrategyGenerator (Claude AI)                        │    │ │
│  │  │  - generate_race_strategy()   - analyze_what_if()                  │    │ │
│  │  │  - compare_strategies()       - explain_strategy_decision()        │    │ │
│  │  │  - analyze_completed_race()   - chat()                             │    │ │
│  │  └────────────────────────────────────────────────────────────────────┘    │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────┘
                                                   │
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              API SERVICE LAYER                                    │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                          fastapi_backend.py                                  │ │
│  │                                                                              │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │ │
│  │  │ Race Data    │  │  Strategy    │  │  Predictions │  │   Search     │    │ │
│  │  │ Endpoints    │  │  Endpoints   │  │  Endpoints   │  │  Endpoints   │    │ │
│  │  │              │  │              │  │              │  │              │    │ │
│  │  │ GET /races   │  │ POST /strategy│ │ POST /predict│  │ POST /query  │    │ │
│  │  │ GET /race/..│  │ /generate    │  │ /position    │  │ GET /search  │    │ │
│  │  │ POST /data/ │  │ POST /what-if│  │ POST /pit-stop│ │ /circuit     │    │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────┘
                                                   │
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT APPLICATIONS                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Web Dashboard│  │  Mobile App  │  │  CLI Tools   │  │  Third-Party         │ │
│  │  (Future)    │  │  (Future)    │  │  (curl/httpie)│  │  Integrations       │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Collection Layer (`data_collection.py`)

#### Purpose
Aggregates F1 data from multiple sources into a unified format.

#### Data Sources

| Source | Data Type | Use Case |
|--------|-----------|----------|
| Ergast API | Historical race data (1950-2024) | Training data, historical context |
| FastF1 | Telemetry, detailed timing | Advanced features, real analysis |
| OpenF1 | Real-time race data | Live predictions (future) |

#### Key Classes

```python
class ErgastAPI:
    """RESTful client for Ergast F1 API"""
    - get_season_schedule(year)
    - get_race_results(year, round)
    - get_pit_stops(year, round)
    - get_lap_times(year, round)
    - get_driver_standings(year)
    
class FastF1Collector:
    """FastF1 library wrapper for telemetry"""
    - get_session(year, race, session_type)
    - get_telemetry(session, driver)
    - analyze_tire_degradation(session, driver)
    
class F1DataCollector:
    """Main collector combining all sources"""
    - collect_race_data(year, round)
    - get_seasons_data(start_year, end_year)
    - get_historical_strategies(circuit_id)
```

### 2. Feature Engineering Layer (`feature_engineering.py`)

#### Purpose
Transforms raw data into ML-ready features with domain knowledge encoded.

#### Feature Categories

| Category | Features | Dimension |
|----------|----------|-----------|
| Circuit | Type, length, corners, DRS zones, tire wear | 12 |
| Tire | Compound, grip, durability, optimal temp | 8 |
| Weather | Track temp, air temp, humidity, rain | 6 |
| Driver | Performance stats, experience | 10 |
| Lap | Progress, position, time, tire age | 6 |

#### Key Processing

```python
# Sequence preparation for LSTM
sequence = prepare_race_sequence(race_data, driver_id)
# Output: (sequence_length, num_features) = (10, 30)

# Training data preparation
X, y, metadata = prepare_training_data(races_data)
# Output: X shape = (n_samples, 10, 30), y shape = (n_samples,)

# Strategy embedding for vector DB
features = create_strategy_embedding_features(race_data, driver_id)
# Output: Dict with text description + metadata
```

### 3. ML Models Layer (`tensorflow_models.py`)

#### RaceOutcomePredictor (LSTM)

```
Input Layer: (10, 30)
    │
    ▼
LSTM Layer 1: 128 units, return_sequences=True
    │
    ▼
BatchNorm + Dropout (0.3)
    │
    ▼
LSTM Layer 2: 64 units
    │
    ▼
BatchNorm + Dropout (0.3)
    │
    ▼
Dense: 64 → 32 (ReLU)
    │
    ├──────────────────────┐
    ▼                      ▼
Dense: 1 (Sigmoid)    Dense: 20 (Softmax)
Position Output       Probability Distribution
```

#### PitStopOptimizer (DQN)

```
State (15 dims):
- Race progress, remaining laps
- Position (normalized)
- Tire age, degradation trend
- Pit stops done
- Circuit/weather features
- Current tire performance

Actions:
- 0: Stay out
- 1: Pit for SOFT
- 2: Pit for MEDIUM
- 3: Pit for HARD

Architecture:
Input → Dense(128) → Dense(128) → Dense(64)
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
         Value Stream                               Advantage Stream
         Dense(32) → Dense(1)                       Dense(32) → Dense(4)
              │                                               │
              └───────────────────┬───────────────────────────┘
                                  ▼
                          Q-Values (Dueling DQN)
```

#### TireDegradationModel

```
Input (8 features):
- Lap in stint
- Fuel load
- Track temp, air temp
- Tire grip, durability
- Average speed
- Cornering intensity

Output:
- Predicted lap time
- Degradation rate
```

### 4. Vector Database Layer (`vector_database.py`)

#### Purpose
Enables semantic search over historical race strategies using embeddings.

#### Architecture

```
Strategy Text → Embedding → Pinecone Index
     │              │              │
     │              │              │
     ▼              ▼              ▼
"2024 Monaco GP,  [0.12, -0.34,   Index: f1-strategies
 Verstappen P1,    0.56, ...]     Dimension: 1536
 1-stop strategy"  1536-dim       Metric: cosine
```

#### Search Flow

```python
# 1. User query
query = "Best wet weather strategy for Spa"

# 2. Generate query embedding
embedding = embedding_generator.generate_embedding(query)

# 3. Search similar strategies
results = pinecone_index.query(
    vector=embedding,
    top_k=5,
    include_metadata=True
)

# 4. Return ranked results
for match in results:
    print(f"{match.metadata['race_name']}: {match.score}")
```

### 5. LLM Strategy Layer (`llm_strategy_generator.py`)

#### RAG Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                      RAG (Retrieval Augmented Generation)        │
│                                                                  │
│  1. Race Info Input                                             │
│     └─→ Circuit: Monaco, Weather: Dry, Laps: 78                 │
│                                                                  │
│  2. Retrieve Historical Context (Vector DB)                     │
│     └─→ Search: "Monaco race strategies"                        │
│     └─→ Results: 5 similar past races                           │
│                                                                  │
│  3. Get ML Predictions                                          │
│     └─→ Position: P3 (72% confidence)                           │
│     └─→ Optimal pit: Lap 25-30                                  │
│                                                                  │
│  4. Construct Prompt                                            │
│     └─→ System: F1 strategist role                              │
│     └─→ Context: Historical + Predictions                       │
│     └─→ Request: Generate strategy                              │
│                                                                  │
│  5. Claude API Call                                             │
│     └─→ Model: claude-sonnet-4-20250514                                │
│     └─→ Response: Structured strategy JSON                      │
│                                                                  │
│  6. Parse and Return                                            │
│     └─→ StrategyResponse object                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6. API Layer (`fastapi_backend.py`)

#### Endpoint Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | System health check |
| `/races` | GET | List season races |
| `/race/{season}/{round}` | GET | Race details |
| `/strategy/generate` | POST | Generate strategy |
| `/strategy/what-if` | POST | Scenario analysis |
| `/query` | POST | Semantic search |
| `/predict/position` | POST | Position prediction |
| `/predict/pit-stop` | POST | Pit recommendation |

## Data Flow Examples

### Strategy Generation Flow

```
1. POST /strategy/generate
   {
     "race_info": {
       "circuit": "monaco",
       "weather": "Dry",
       "total_laps": 78
     }
   }
   
2. AppState initialization check
   - data_collector ✓
   - vector_db ✓
   - strategy_generator ✓
   - ml_models ✓

3. Vector DB search
   query = "Race strategy for monaco"
   historical_context = [5 similar strategies]

4. ML predictions (if models loaded)
   position_pred = RaceOutcomePredictor.predict()
   pit_recommendation = PitStopOptimizer.get_optimal_strategy()

5. Claude API call
   prompt = format_strategy_prompt(race_info, historical, ml_pred)
   response = claude.messages.create(prompt)

6. Parse and return StrategyResponse
   {
     "executive_summary": "...",
     "recommended_strategy": {...},
     "alternative_strategies": [...],
     "confidence_score": 0.85
   }
```

## Training Pipeline

```
train_models.py
     │
     ├── 1. collect_training_data()
     │   └── F1DataCollector.get_seasons_data(2015, 2024)
     │
     ├── 2. prepare_features()
     │   └── F1FeatureEngineer.prepare_training_data()
     │
     ├── 3. train_race_predictor()
     │   ├── RaceOutcomePredictor.build_model()
     │   ├── RaceOutcomePredictor.train()
     │   └── Save to ./models/race_predictor.keras
     │
     ├── 4. train_pit_stop_optimizer()
     │   ├── PitStopOptimizer (DQN)
     │   ├── Simulate 500 episodes
     │   └── Save to ./models/pit_optimizer/
     │
     ├── 5. train_tire_degradation_model()
     │   ├── TireDegradationModel.train()
     │   └── Save to ./models/degradation_model.keras
     │
     └── 6. initialize_vector_database()
         ├── StrategyIndexer.index_race_strategies()
         └── Pinecone index populated
```

## Deployment Configurations

### Development
- Single container
- Hot reload enabled
- Debug logging
- Local file storage

### Production
- Multi-container (Docker Compose)
- Load balanced
- Redis caching
- S3/DynamoDB storage
- Prometheus/Grafana monitoring

### Serverless (AWS)
- Lambda functions
- API Gateway
- EventBridge triggers
- S3 + DynamoDB storage

## Technology Stack Summary

| Layer | Technology |
|-------|------------|
| Language | Python 3.9+ |
| ML Framework | TensorFlow 2.15 |
| API Framework | FastAPI |
| Vector DB | Pinecone |
| LLM | Claude (Anthropic) |
| Data Sources | Ergast, FastF1, OpenF1 |
| Containerization | Docker |
| Cloud | AWS (Lambda, S3, DynamoDB) |
| Monitoring | Prometheus + Grafana |
