# 🚀 F1 Fantasy - Quick Start

## TL;DR

Your F1 fantasy system is ready! Run one command to get started:

```bash
python ingest_fantasy_data.py
```

## What This Gives You

🎯 **Driver Recommendations** - Who to pick for each circuit
📊 **Circuit Analysis** - Overtaking potential, qualifying importance
🔥 **Recent Form** - Momentum tracking for hot/cold drivers
👥 **Head-to-Head** - Teammate and rival comparisons
💰 **Value Picks** - Consistent scorers for budget optimization

## One-Command Setup

```bash
# Recommended: Last 4 years (2020-2024)
python ingest_fantasy_data.py

# Time: ~10-15 minutes
# Cost: ~$1.50
# Vectors: ~2,600
```

## Test It Works

```python
from src.vector_db import F1VectorDB

vdb = F1VectorDB()

# Ask: "Who should I pick for Monaco?"
results = vdb.search_similar_races(
    query="Monaco Circuit best performers",
    top_k=5
)

print(f"Top 5 picks for Monaco:")
for r in results:
    print(f"  • {r['metadata']['driver_name']}")
```

## Try the Examples

### API (FastAPI)
```bash
pip install fastapi uvicorn
uvicorn examples.fantasy_api_example:app --reload
# Visit: http://localhost:8000/docs
```

### Dashboard (Streamlit)
```bash
pip install streamlit
streamlit run examples/fantasy_dashboard.py
# Visit: http://localhost:8501
```

## Key Features

### 1. Circuit-Specific Recommendations
```python
# "Who's good at Monaco?"
vdb.search_similar_races(
    query="Monaco Circuit",
    filter_dict={'type': 'fantasy_driver_circuit'}
)
```

### 2. Recent Form Analysis
```python
# "Who's hot right now?"
vdb.search_similar_races(
    query="improving momentum good recent form",
    filter_dict={'type': 'fantasy_driver_profile'}
)
```

### 3. Circuit Characteristics
```python
# "Is qualifying important here?"
vdb.search_similar_races(
    query="Circuit characteristics",
    filter_dict={'type': 'fantasy_circuit_analysis'}
)
```

### 4. Head-to-Head
```python
# "Verstappen vs Perez?"
vdb.search_similar_races(
    query="VER vs PER",
    filter_dict={'type': 'fantasy_head_to_head'}
)
```

## Example Queries

```python
# Value picks
vdb.search_similar_races(
    query="consistent points scorer low DNF rate",
    top_k=10
)

# Overtakers
vdb.search_similar_races(
    query="gains positions race pace overtaking",
    top_k=10
)

# Qualifiers
vdb.search_similar_races(
    query="strong qualifier grid position",
    top_k=10
)
```

## Build Your App

### Basic Recommendation Flow

```python
def recommend_for_circuit(circuit_name):
    vdb = F1VectorDB()

    # 1. Analyze circuit
    circuit = vdb.search_similar_races(
        query=f"{circuit_name} characteristics",
        filter_dict={'type': 'fantasy_circuit_analysis'}
    )[0]

    # 2. Choose strategy
    if "qualifying" in circuit['text'] and "very high" in circuit['text']:
        query = "strong qualifier"
    else:
        query = "gains positions overtaking"

    # 3. Get recommendations
    drivers = vdb.search_similar_races(
        query=f"{circuit_name} {query}",
        top_k=10,
        filter_dict={'type': 'fantasy_driver_circuit'}
    )

    return drivers
```

## API Endpoints (Examples)

```bash
# Recommendations
GET /fantasy/recommend?circuit=Monaco&strategy=qualifiers

# Circuit analysis
GET /fantasy/circuit-analysis?circuit=Monaco

# Recent form
GET /fantasy/recent-form?momentum=improving

# Natural language
GET /fantasy/ask?question=Who should I pick for Monaco?
```

## Files

**Core:**
- [src/fantasy_data_ingestion.py](src/fantasy_data_ingestion.py)
- [src/vector_db.py](src/vector_db.py)

**Scripts:**
- [ingest_fantasy_data.py](ingest_fantasy_data.py) - **Run this first!**

**Examples:**
- [examples/fantasy_api_example.py](examples/fantasy_api_example.py)
- [examples/fantasy_dashboard.py](examples/fantasy_dashboard.py)

**Docs:**
- [FANTASY_READY.md](FANTASY_READY.md) - Complete overview
- [FANTASY_GUIDE.md](FANTASY_GUIDE.md) - Detailed guide

## Next Steps

1. ✅ Run: `python ingest_fantasy_data.py`
2. ✅ Test examples
3. ✅ Build your customer app
4. ✅ Launch! 🚀

## Questions?

Check the logs: `logs/fantasy_ingestion.log`

Test the database:
```python
stats = vdb.get_stats()
print(f"Vectors: {stats['total_vectors']:,}")
```

**You're ready to build your F1 fantasy app!** 🏎️🏆
