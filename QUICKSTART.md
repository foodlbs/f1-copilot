# Quick Start Guide

Get up and running with the F1 Knowledge Base in 5 minutes.

## 1. Prerequisites

```bash
# Check Python version (requires 3.9+)
python --version

# Install pip if needed
python -m ensurepip --upgrade
```

## 2. Get API Keys

### Pinecone (Vector Database)
1. Go to [pinecone.io](https://www.pinecone.io/)
2. Sign up for free account
3. Create new project
4. Copy API key from dashboard

### OpenAI (Embeddings)
1. Go to [platform.openai.com](https://platform.openai.com/)
2. Sign up or log in
3. Go to API keys section
4. Create new API key
5. Copy the key (shown only once!)

## 3. Install

```bash
# Clone repository
git clone <your-repo-url>
cd BuildWatch

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 4. Configure

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
# Use nano, vim, or your preferred editor
nano .env
```

Add your keys to `.env`:
```bash
PINECONE_API_KEY=your_pinecone_key_here
OPENAI_API_KEY=your_openai_key_here
START_YEAR=2017
```

## 5. Build Knowledge Base

```bash
cd src
python knowledge_base_builder.py
```

This will:
- Download F1 race data from 2017-present (~2-4 hours)
- Generate embeddings
- Create vector database

**Coffee break recommended!** ☕

## 6. Test Strategy Prediction

```bash
cd src
python strategy_predictor.py
```

You should see strategy recommendations for Monza.

## 7. Run Race Simulation

```bash
cd src
python race_simulator.py
```

Watch a simulated race unfold lap-by-lap!

## Quick Examples

### Predict Strategy

```python
from src.strategy_predictor import F1StrategyPredictor, SessionData

predictor = F1StrategyPredictor()

session = SessionData(
    circuit="Spa",
    session_type="Race",
    lap_number=1,
    total_laps=44,
    air_temp=20.0,
    track_temp=28.0,
    weather="Dry",
    available_compounds=["SOFT", "MEDIUM", "HARD"]
)

recommendation = predictor.predict_optimal_strategy(session)
print(f"Strategy: {recommendation.strategy_type}")
```

### Search Similar Races

```python
from src.vector_db import F1VectorDB

vdb = F1VectorDB()

results = vdb.search_similar_races(
    query="High-speed circuit dry conditions",
    top_k=5
)

for r in results:
    print(f"{r['metadata']['race_name']} - Score: {r['score']:.3f}")
```

## Troubleshooting

### "Pinecone API key not found"
- Check `.env` file exists
- Verify API key is correct
- Try `source .env` before running

### "FastF1 download failed"
- Check internet connection
- Some races may have limited data
- Script will continue with available data

### "Out of memory"
- Reduce `START_YEAR` in `.env` (e.g., 2020)
- Process fewer years at once
- Increase system swap space

### "OpenAI rate limit"
- Free tier has limits
- Script will retry automatically
- Consider upgrading plan for faster processing

## Next Steps

1. Read full [README.md](README.md) for detailed documentation
2. Explore the code in `src/` directory
3. Customize for your use case
4. Build integrations (APIs, dashboards, etc.)

## Tips

- Start with recent years (2020+) for faster initial build
- Use `skip_ingestion=True` to collect data without vector DB
- Cache is stored in `cache/` - don't delete during builds
- Check `logs/` for detailed progress

## Getting Help

- Check documentation in [README.md](README.md)
- Review code comments in source files
- Open issue on GitHub

Happy racing! 🏎️💨
