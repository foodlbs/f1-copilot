# 🏎️ F1 Race Strategy Analyzer - Quick Start Guide

Get up and running with the F1 Race Strategy Analyzer in 15 minutes!

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git (optional, for cloning)
- API Keys:
  - [Pinecone](https://www.pinecone.io/) - Vector database (free tier available)
  - [Anthropic](https://console.anthropic.com/) - Claude AI (pay-per-use)

## 🚀 Option 1: Local Development (Recommended for Learning)

### Step 1: Setup Environment

```bash
# Create project directory
cd f1-race-analyzer

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure API Keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# PINECONE_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here
```

### Step 3: Initialize the System

```python
# Run this Python script to initialize
python << 'EOF'
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize vector database
from vector_database import F1VectorDatabase
vdb = F1VectorDatabase()
vdb.create_index()
print("✅ Vector database initialized")

# Collect sample data
from data_collection import F1DataCollector
collector = F1DataCollector(use_fastf1=False)

# Get 2024 season schedule
schedule = collector.ergast.get_season_schedule(2024)
print(f"✅ Found {len(schedule)} races in 2024 season")

# Collect one race as sample
if schedule:
    race_data = collector.collect_race_data(2024, 1)
    print(f"✅ Collected data for: {race_data.race_name}")

print("\n🎉 System initialized successfully!")
EOF
```

### Step 4: Start the API Server

```bash
# Start the FastAPI server
python fastapi_backend.py
```

Visit http://localhost:8000/docs for interactive API documentation!

### Step 5: Try It Out!

```bash
# Generate a race strategy
curl -X POST http://localhost:8000/strategy/generate \
  -H "Content-Type: application/json" \
  -d '{
    "race_info": {
      "circuit": "monaco",
      "weather_forecast": "Dry, 24C",
      "total_laps": 78,
      "driver": "Max Verstappen",
      "grid_position": 1
    }
  }'

# Search historical strategies
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Best tire strategy for Monaco wet conditions",
    "top_k": 5
  }'
```

---

## 🐳 Option 2: Docker (Fastest Setup)

### Step 1: Configure Environment

```bash
# Create .env file
cat > .env << EOF
PINECONE_API_KEY=your_pinecone_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
EOF
```

### Step 2: Start Services

```bash
# Build and start the API
docker-compose up -d api

# Check logs
docker-compose logs -f api

# Verify it's running
curl http://localhost:8000/health
```

### Step 3: (Optional) Run Training

```bash
# Quick training (5-10 minutes)
docker-compose --profile training-quick up trainer-quick

# Full training (30-60 minutes)
docker-compose --profile training up trainer
```

---

## 🧪 Quick Test

Run this Python script to test all components:

```python
# test_setup.py
import os
from dotenv import load_dotenv
load_dotenv()

print("🧪 Testing F1 Race Strategy Analyzer Setup\n")

# Test 1: Data Collection
print("1️⃣ Testing Data Collection...")
try:
    from data_collection import F1DataCollector
    collector = F1DataCollector(use_fastf1=False)
    schedule = collector.ergast.get_season_schedule(2024)
    print(f"   ✅ Data collection works! Found {len(schedule)} races\n")
except Exception as e:
    print(f"   ❌ Data collection failed: {e}\n")

# Test 2: Feature Engineering
print("2️⃣ Testing Feature Engineering...")
try:
    from feature_engineering import F1FeatureEngineer
    engineer = F1FeatureEngineer()
    print("   ✅ Feature engineering works!\n")
except Exception as e:
    print(f"   ❌ Feature engineering failed: {e}\n")

# Test 3: TensorFlow Models
print("3️⃣ Testing TensorFlow Models...")
try:
    from tensorflow_models import RaceOutcomePredictor, PitStopOptimizer
    predictor = RaceOutcomePredictor()
    predictor.build_model()
    optimizer = PitStopOptimizer()
    print("   ✅ TensorFlow models work!\n")
except Exception as e:
    print(f"   ❌ TensorFlow models failed: {e}\n")

# Test 4: Vector Database
print("4️⃣ Testing Vector Database...")
try:
    from vector_database import F1VectorDatabase
    vdb = F1VectorDatabase()
    vdb.create_index()
    stats = vdb.get_index_stats()
    print(f"   ✅ Vector database works! Vectors: {stats.get('total_vectors', 0)}\n")
except Exception as e:
    print(f"   ❌ Vector database failed: {e}\n")

# Test 5: Strategy Generator
print("5️⃣ Testing Strategy Generator...")
try:
    from llm_strategy_generator import F1StrategyGenerator
    generator = F1StrategyGenerator()
    print("   ✅ Strategy generator initialized!\n")
except Exception as e:
    print(f"   ❌ Strategy generator failed: {e}\n")

print("🏁 Setup test complete!")
```

---

## 📊 Training Models

Train all ML models on historical data:

```bash
# Quick training (for testing, ~5 minutes)
python train_models.py --quick

# Full training (~30-60 minutes)
python train_models.py --start-year 2015 --end-year 2024 --epochs 100

# Training with vector database initialization
python train_models.py --quick --init-vectordb
```

---

## 🎯 Example Usage

### Generate a Race Strategy

```python
from llm_strategy_generator import F1StrategyGenerator, StrategyRequest
import os
from dotenv import load_dotenv
load_dotenv()

generator = F1StrategyGenerator()

race_info = {
    'circuit': 'Silverstone',
    'weather_forecast': 'Dry, 22°C, 30% chance of rain',
    'total_laps': 52,
    'driver': 'Lewis Hamilton',
    'constructor': 'Mercedes',
    'grid_position': 3
}

strategy = generator.generate_race_strategy(race_info)

print("📋 Executive Summary:")
print(strategy.executive_summary)

print("\n🎯 Recommended Strategy:")
print(strategy.recommended_strategy)
```

### Search Historical Strategies

```python
from vector_database import F1VectorDatabase

db = F1VectorDatabase()
db.create_index()

results = db.search_similar_strategies(
    query="Wet weather strategy for Spa with safety car",
    top_k=5
)

for result in results:
    print(f"- {result['metadata'].get('race_name')}: Score {result['score']:.2f}")
```

### Get ML Predictions

```python
import numpy as np
from tensorflow_models import RaceOutcomePredictor

# Load trained model
predictor = RaceOutcomePredictor()
predictor.load_model('models/race_predictor.keras')

# Sample input (your actual feature data)
race_sequence = np.random.randn(1, 10, 30)

# Predict
position, probabilities = predictor.predict_position(race_sequence)
print(f"Predicted position: P{position}")
print(f"Top 3 probability: {probabilities[:3].sum():.1%}")
```

---

## 🔧 Troubleshooting

### Common Issues

**1. "Module not found" errors**
```bash
# Ensure you're in the virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt
```

**2. API key errors**
```bash
# Verify your .env file exists and has correct keys
cat .env

# Test the keys
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('Pinecone:', 'SET' if os.getenv('PINECONE_API_KEY') else 'MISSING')"
```

**3. TensorFlow GPU issues**
```bash
# Use CPU-only TensorFlow if GPU causes issues
pip uninstall tensorflow
pip install tensorflow-cpu
```

**4. Docker issues**
```bash
# Rebuild containers
docker-compose build --no-cache

# Check logs
docker-compose logs api
```

---

## 📚 Next Steps

1. **Explore the API**: Visit http://localhost:8000/docs
2. **Train models**: Run `python train_models.py`
3. **Read the full docs**: See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
4. **Customize**: Modify prompts in `llm_strategy_generator.py`
5. **Add data**: Collect more seasons with `data_collection.py`

---

## 🆘 Getting Help

- Check the [README.md](./README.md) for full documentation
- Open an issue on GitHub
- Review the code comments in each module

Happy racing! 🏎️
