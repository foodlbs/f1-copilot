# F1 Fantasy Lineup Assistant - Complete Guide

## Overview

This enhanced ingestion system creates a vector database optimized for F1 fantasy lineup recommendations. It goes beyond basic race results to include performance trends, consistency metrics, circuit-specific analysis, and head-to-head comparisons.

## What Makes This Fantasy-Optimized?

### 1. Driver Performance Metrics

For each driver, the system tracks:

- **Average Finish Position**: Season-long performance
- **Points Per Race**: Fantasy value metric
- **Podiums & Top 10s**: Consistency indicators
- **Position Changes**: Overtaking ability (grid → finish)
- **DNF Rate**: Reliability risk factor
- **Consistency Score**: Standard deviation of finishes
- **Grid vs Finish**: Qualifying-to-race conversion

### 2. Recent Form & Momentum

**Last 5 races analysis:**
- Average finishing position
- Average points scored
- **Momentum Score**: Linear regression on positions
  - Positive = improving (gaining positions)
  - Negative = declining
- **Improving Flag**: Boolean for quick filtering

Perfect for: "Show me drivers with positive momentum going into this weekend"

### 3. Circuit-Specific Performance

For each driver at each circuit:
- Historical races at circuit
- Average finish position
- Best finish ever
- Average points scored
- Last 3 races average (recent circuit form)

Perfect for: "Who performs best at Monaco?" or "Show me Max's history at Monza"

### 4. Circuit Characteristics

For each circuit, analyzes:

**Overtaking Potential:**
- High overtaking (avg 3+ position changes)
- Moderate (1.5-3 changes)
- Low (< 1.5 changes)

**Qualifying Importance:**
- Very high: Top 3 quali → avg P3.5 finish
- High: Top 3 quali → avg P5 finish
- Moderate: Top 3 quali → avg P6+ finish

**Strategy:**
- Typical number of pit stops
- Strategy variation

**Reliability Risk:**
- Historical DNF rate at circuit
- High/moderate/low risk classification

Perfect for: "Is qualifying critical this weekend?" or "Should I pick overtakers or qualifiers?"

### 5. Head-to-Head Comparisons

Direct teammate and competitor comparisons:
- Win percentage when both finish
- Total head-to-head races
- Useful for: "Perez vs Verstappen at Red Bull"

### 6. Constructor Reliability

- Average points per race
- Reliability rate (finishes / total)
- Total DNFs
- Trend analysis

## Quick Start

### Install & Setup

Already done! Your system is configured from the base CSV ingestion.

### Ingest Fantasy Data

```bash
# Recommended: Recent era (2020-2024)
python ingest_fantasy_data.py

# Just 2024 season
python ingest_fantasy_data.py --year 2024

# Custom range
python ingest_fantasy_data.py --start-year 2018 --end-year 2023

# Skip head-to-head (faster ingestion)
python ingest_fantasy_data.py --no-head-to-head
```

## What Gets Created

### Document Types

#### 1. Driver Performance Profiles
**One per driver** with overall metrics:
```
Driver: Max Verstappen (VER) | Team: Red Bull Racing |
Season avg finish: P1.8 | Avg points per race: 23.4 |
Podiums: 18, Top 10s: 22 | Position changes avg: +1.2 |
DNF rate: 4.3% | Consistency: 0.87 |
Recent form (last 5): Avg P1.4, 24.8 pts/race |
Momentum: improving (score: +0.6)
```

**Metadata:**
```json
{
  "type": "fantasy_driver_profile",
  "driver_code": "VER",
  "driver_name": "Max Verstappen",
  "team": "Red Bull Racing",
  "data_type": "overall_performance"
}
```

#### 2. Driver Circuit Performance
**One per driver per circuit:**
```
Driver: Lewis Hamilton (HAM) | Team: Mercedes |
Season avg finish: P4.2 | Avg points per race: 12.3 |
At Silverstone: 15 races, avg P2.1, best P1 |
Recent form (last 5): Avg P3.8, 13.2 pts/race
```

**Metadata:**
```json
{
  "type": "fantasy_driver_circuit",
  "driver_code": "HAM",
  "driver_name": "Lewis Hamilton",
  "circuit": "Silverstone Circuit",
  "data_type": "circuit_performance"
}
```

#### 3. Circuit Analysis
**One per circuit:**
```
Circuit: Monaco Circuit | Historical races: 74 |
Overtaking: low overtaking (avg 0.8 position changes) |
Typical strategy: 1.2 stop(s) |
Qualifying importance: very high (top 3 quali avg finish P2.4) |
Circuit reliability: moderate risk (9.2% DNF rate)
```

**Metadata:**
```json
{
  "type": "fantasy_circuit_analysis",
  "circuit": "Monaco Circuit",
  "data_type": "circuit_characteristics"
}
```

#### 4. Head-to-Head Comparisons
**For teammates and close competitors:**
```
Head-to-head: Sergio Pérez vs Max Verstappen |
Sergio Pérez: 8 wins (23%) | Max Verstappen: 27 wins (77%) |
Total comparisons: 35
```

**Metadata:**
```json
{
  "type": "fantasy_head_to_head",
  "driver1": "PER",
  "driver2": "VER",
  "data_type": "head_to_head_comparison"
}
```

## Using the Data for Fantasy Recommendations

### Example Queries

#### 1. Find Value Picks
```python
from src.vector_db import F1VectorDB

vdb = F1VectorDB()

# High points per race, low DNF rate
results = vdb.search_similar_races(
    query="driver consistent points scorer low DNF rate improving momentum",
    top_k=10,
    filter_dict={'type': 'fantasy_driver_profile'}
)

for r in results:
    print(f"{r['score']:.3f} - {r['metadata']['driver_name']}")
```

#### 2. Circuit-Specific Recommendations
```python
# Monaco this weekend - who's good there?
results = vdb.search_similar_races(
    query="Monaco street circuit best historical performance",
    top_k=5,
    filter_dict={'type': 'fantasy_driver_circuit', 'circuit': 'Monaco Circuit'}
)
```

#### 3. Qualifying vs Race Pace
```python
# Circuit where qualifying matters
circuit_info = vdb.search_similar_races(
    query="qualifying important track position critical",
    top_k=1,
    filter_dict={'type': 'fantasy_circuit_analysis'}
)

# If qualifying is critical, find strong qualifiers
if "very high" in circuit_info[0]['text']:
    drivers = vdb.search_similar_races(
        query="strong qualifier good grid position",
        top_k=10,
        filter_dict={'type': 'fantasy_driver_profile'}
    )
```

#### 4. Find Overtakers
```python
# Bahrain - high overtaking circuit
# Find drivers who gain positions
results = vdb.search_similar_races(
    query="driver gains positions overtaking race pace",
    top_k=10,
    filter_dict={'type': 'fantasy_driver_profile'}
)
```

#### 5. Recent Form
```python
# Hot drivers right now
results = vdb.search_similar_races(
    query="improving momentum recent form good last 5 races",
    top_k=10,
    filter_dict={'type': 'fantasy_driver_profile'}
)
```

#### 6. Teammate Battles
```python
# Compare teammates
results = vdb.search_similar_races(
    query="Perez vs Verstappen Red Bull teammate",
    top_k=1,
    filter_dict={'type': 'fantasy_head_to_head'}
)
```

## Building a Fantasy Assistant App

### Example Architecture

```python
class F1FantasyAssistant:
    def __init__(self):
        self.vdb = F1VectorDB()

    def recommend_lineup(
        self,
        circuit: str,
        budget: float,
        avoid_dnf_risk: bool = True
    ) -> List[Dict]:
        """Generate fantasy lineup recommendation"""

        # Step 1: Analyze circuit characteristics
        circuit_analysis = self.vdb.search_similar_races(
            query=f"{circuit} circuit characteristics overtaking qualifying",
            top_k=1,
            filter_dict={'type': 'fantasy_circuit_analysis', 'circuit': circuit}
        )[0]

        # Step 2: Determine strategy (qualifiers vs overtakers)
        if "qualifying importance: very high" in circuit_analysis['text']:
            strategy = "strong qualifiers low DNF rate"
        elif "high overtaking" in circuit_analysis['text']:
            strategy = "gains positions race pace overtaking"
        else:
            strategy = "consistent points scorer good form"

        # Step 3: Find circuit-specific performers
        circuit_specialists = self.vdb.search_similar_races(
            query=f"{circuit} best historical performance {strategy}",
            top_k=20,
            filter_dict={'type': 'fantasy_driver_circuit', 'circuit': circuit}
        )

        # Step 4: Check recent form
        in_form_drivers = self.vdb.search_similar_races(
            query="improving momentum recent form good last 5 races",
            top_k=20,
            filter_dict={'type': 'fantasy_driver_profile'}
        )

        # Step 5: Combine scores and optimize for budget
        recommendations = self._optimize_lineup(
            circuit_specialists,
            in_form_drivers,
            budget,
            avoid_dnf_risk
        )

        return recommendations

    def explain_pick(self, driver_code: str, circuit: str) -> str:
        """Explain why a driver is a good pick"""

        # Get driver profile
        profile = self.vdb.search_similar_races(
            query=f"{driver_code} driver performance",
            top_k=1,
            filter_dict={'type': 'fantasy_driver_profile', 'driver_code': driver_code}
        )[0]

        # Get circuit-specific
        circuit_perf = self.vdb.search_similar_races(
            query=f"{driver_code} {circuit}",
            top_k=1,
            filter_dict={'type': 'fantasy_driver_circuit', 'driver_code': driver_code}
        )[0]

        explanation = f"""
        {profile['metadata']['driver_name']} is a solid pick because:

        Overall Form:
        {profile['text']}

        At {circuit}:
        {circuit_perf['text']}
        """

        return explanation
```

### Simple Query Interface

```python
def natural_language_query(user_question: str) -> str:
    """Answer fantasy questions using vector search"""
    vdb = F1VectorDB()

    # Search across all fantasy documents
    results = vdb.search_similar_races(
        query=user_question,
        top_k=5
    )

    # Format response
    context = "\n\n".join([r['text'] for r in results])

    # Use LLM for final answer (with context)
    prompt = f"""
    User question: {user_question}

    Relevant F1 data:
    {context}

    Provide a helpful fantasy recommendation based on this data.
    """

    # Call your LLM here (OpenAI, Claude, etc.)
    return llm_response(prompt)

# Usage
answer = natural_language_query("Who should I pick for Monaco this weekend?")
answer = natural_language_query("Is Verstappen worth his price?")
answer = natural_language_query("Which Ferrari driver is better value?")
```

## Data Volume & Costs

### 2020-2024 (Recommended for Fantasy)

**Base data:**
- ~100 races
- ~2,000 base vectors

**Fantasy additions:**
- ~20 drivers × overall profiles = 20 vectors
- ~20 drivers × 25 circuits = 500 vectors
- ~25 circuits × analysis = 25 vectors
- ~50 head-to-head = 50 vectors

**Total: ~2,600 vectors**

**Cost: ~$1.50** (embeddings)
**Time: ~10-15 minutes**

### 2024 Only (Testing)

**Total: ~700 vectors**
**Cost: ~$0.40**
**Time: ~3-5 minutes**

### Full Modern Era (2010-2024)

**Total: ~8,000 vectors**
**Cost: ~$4.50**
**Time: ~25-30 minutes**

## Performance Tips

### 1. Start with Recent Data

Fantasy is about current form, so 2020-2024 is usually enough:
```bash
python ingest_fantasy_data.py --start-year 2020
```

### 2. Skip Head-to-Head for Speed

If you don't need teammate comparisons:
```bash
python ingest_fantasy_data.py --no-head-to-head
```
Saves ~50% of fantasy document generation time.

### 3. Update Weekly

Before each race weekend:
```bash
# Re-run with latest data
python ingest_fantasy_data.py --start-year 2024
```

This updates recent form and momentum scores.

## API Examples

### FastAPI Endpoint

```python
from fastapi import FastAPI
from src.vector_db import F1VectorDB

app = FastAPI()
vdb = F1VectorDB()

@app.get("/fantasy/recommend")
def recommend_drivers(
    circuit: str,
    position_type: str = "any"  # qualifier, overtaker, consistent
):
    """Get driver recommendations for a circuit"""

    query = f"{circuit} {position_type} driver best performance"

    results = vdb.search_similar_races(
        query=query,
        top_k=10,
        filter_dict={'type': 'fantasy_driver_circuit', 'circuit': circuit}
    )

    return {
        "circuit": circuit,
        "recommendations": [
            {
                "driver": r['metadata']['driver_name'],
                "score": r['score'],
                "analysis": r['text']
            }
            for r in results
        ]
    }

@app.get("/fantasy/circuit-analysis/{circuit}")
def analyze_circuit(circuit: str):
    """Get circuit characteristics for fantasy strategy"""

    results = vdb.search_similar_races(
        query=f"{circuit} overtaking qualifying strategy",
        top_k=1,
        filter_dict={'type': 'fantasy_circuit_analysis', 'circuit': circuit}
    )

    if results:
        analysis = results[0]['text']
        return {
            "circuit": circuit,
            "analysis": analysis,
            "strategy_recommendation": _parse_strategy(analysis)
        }

    return {"error": "Circuit not found"}

def _parse_strategy(analysis_text: str) -> str:
    """Extract fantasy strategy from circuit analysis"""
    if "very high" in analysis_text and "qualifying" in analysis_text:
        return "Focus on strong qualifiers - track position is critical"
    elif "high overtaking" in analysis_text:
        return "Pick drivers with good race pace and overtaking ability"
    elif "low overtaking" in analysis_text:
        return "Qualifying is key - pick drivers who qualify well"
    else:
        return "Balanced strategy - consistency and form matter most"
```

### Streamlit Dashboard

```python
import streamlit as st
from src.vector_db import F1VectorDB

st.title("F1 Fantasy Assistant")

vdb = F1VectorDB()

# Circuit selector
circuit = st.selectbox("Select Circuit", [
    "Monaco Circuit",
    "Silverstone Circuit",
    "Monza Circuit",
    # ... etc
])

# Get circuit analysis
if st.button("Analyze Circuit"):
    analysis = vdb.search_similar_races(
        query=f"{circuit} characteristics",
        top_k=1,
        filter_dict={'type': 'fantasy_circuit_analysis'}
    )[0]

    st.subheader("Circuit Characteristics")
    st.write(analysis['text'])

# Driver recommendations
if st.button("Get Recommendations"):
    drivers = vdb.search_similar_races(
        query=f"{circuit} best performers good form",
        top_k=10,
        filter_dict={'type': 'fantasy_driver_circuit', 'circuit': circuit}
    )

    st.subheader("Recommended Drivers")
    for i, driver in enumerate(drivers, 1):
        with st.expander(f"{i}. {driver['metadata']['driver_name']} - Score: {driver['score']:.3f}"):
            st.write(driver['text'])
```

## Summary

The fantasy-optimized ingestion creates a powerful knowledge base for F1 fantasy recommendations by:

✅ **Performance Metrics**: Overall stats, consistency, reliability
✅ **Recent Form**: Last 5 races momentum analysis
✅ **Circuit-Specific**: Historical performance at each track
✅ **Circuit Analysis**: Overtaking, qualifying importance, strategy
✅ **Head-to-Head**: Teammate and competitor comparisons
✅ **Constructor Data**: Team reliability and performance

All searchable via semantic queries and ready to power a customer-facing fantasy assistant application!

**Next Steps:**
1. Run fantasy ingestion: `python ingest_fantasy_data.py`
2. Test queries for your use cases
3. Build your fantasy recommendation API
4. Create customer-facing UI

Happy fantasy racing! 🏎️🏆
