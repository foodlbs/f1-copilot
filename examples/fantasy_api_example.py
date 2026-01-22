"""
F1 Fantasy API Example

Example FastAPI implementation for F1 fantasy lineup recommendations.
Uses the vector database to provide intelligent driver recommendations.

Run with: uvicorn examples.fantasy_api_example:app --reload
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from vector_db import F1VectorDB
from dotenv import load_dotenv

load_dotenv()

# Initialize
app = FastAPI(
    title="F1 Fantasy API",
    description="AI-powered F1 fantasy lineup recommendations",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vector DB connection
vdb = F1VectorDB()


# Models
class DriverRecommendation(BaseModel):
    driver_name: str
    driver_code: str
    team: str
    relevance_score: float
    analysis: str


class CircuitAnalysis(BaseModel):
    circuit: str
    overtaking_potential: str
    qualifying_importance: str
    typical_stops: str
    dnf_risk: str
    strategy_recommendation: str


class HeadToHeadComparison(BaseModel):
    driver1: str
    driver2: str
    driver1_wins: int
    driver2_wins: int
    driver1_win_pct: float
    total_races: int


# Endpoints

@app.get("/")
def root():
    """API root endpoint"""
    stats = vdb.get_stats()
    return {
        "message": "F1 Fantasy API",
        "version": "1.0.0",
        "status": "operational",
        "vectors_indexed": stats['total_vectors']
    }


@app.get("/fantasy/recommend", response_model=List[DriverRecommendation])
def recommend_drivers(
    circuit: str = Query(..., description="Circuit name (e.g., 'Monaco Circuit')"),
    strategy: str = Query("balanced", description="Strategy: 'qualifiers', 'overtakers', or 'balanced'"),
    limit: int = Query(10, ge=1, le=20, description="Number of recommendations")
):
    """
    Get driver recommendations for a specific circuit

    Args:
        circuit: The circuit name
        strategy: Focus on qualifiers, overtakers, or balanced
        limit: Number of drivers to recommend
    """

    # Build query based on strategy
    if strategy == "qualifiers":
        query = f"{circuit} strong qualifier good grid position qualifying pace"
    elif strategy == "overtakers":
        query = f"{circuit} gains positions overtaking race pace improving positions"
    else:
        query = f"{circuit} consistent good form best performers"

    # Search for circuit-specific recommendations
    results = vdb.search_similar_races(
        query=query,
        top_k=limit,
        filter_dict={'type': 'fantasy_driver_circuit', 'circuit': circuit}
    )

    if not results:
        raise HTTPException(status_code=404, detail=f"No data found for circuit: {circuit}")

    recommendations = []
    for result in results:
        recommendations.append(DriverRecommendation(
            driver_name=result['metadata']['driver_name'],
            driver_code=result['metadata']['driver_code'],
            team=result['metadata']['team'],
            relevance_score=round(result['score'], 3),
            analysis=result['text']
        ))

    return recommendations


@app.get("/fantasy/circuit-analysis", response_model=CircuitAnalysis)
def analyze_circuit(
    circuit: str = Query(..., description="Circuit name")
):
    """
    Get detailed circuit analysis for fantasy strategy

    Returns overtaking potential, qualifying importance, and recommended strategy.
    """

    results = vdb.search_similar_races(
        query=f"{circuit} characteristics overtaking qualifying strategy",
        top_k=1,
        filter_dict={'type': 'fantasy_circuit_analysis', 'circuit': circuit}
    )

    if not results:
        raise HTTPException(status_code=404, detail=f"Circuit not found: {circuit}")

    analysis_text = results[0]['text']

    # Parse characteristics
    def extract_characteristic(text: str, keyword: str) -> str:
        """Extract characteristic from analysis text"""
        for line in text.split('|'):
            if keyword in line.lower():
                return line.strip()
        return "Not available"

    overtaking = extract_characteristic(analysis_text, "overtaking")
    qualifying = extract_characteristic(analysis_text, "qualifying")
    stops = extract_characteristic(analysis_text, "strategy")
    dnf = extract_characteristic(analysis_text, "reliability")

    # Generate strategy recommendation
    if "very high" in qualifying.lower():
        strategy = "🎯 Qualifying is CRITICAL - Focus on strong qualifiers. Track position will be hard to change."
    elif "high overtaking" in overtaking.lower():
        strategy = "🏁 Overtaking circuit - Pick drivers with strong race pace. Starting position less critical."
    elif "low overtaking" in overtaking.lower():
        strategy = "⚠️ Low overtaking - Qualifying matters a lot. Pick good qualifiers."
    else:
        strategy = "⚖️ Balanced circuit - Focus on consistency and recent form."

    if "high risk" in dnf.lower():
        strategy += " Consider DNF risk when choosing drivers."

    return CircuitAnalysis(
        circuit=circuit,
        overtaking_potential=overtaking,
        qualifying_importance=qualifying,
        typical_stops=stops,
        dnf_risk=dnf,
        strategy_recommendation=strategy
    )


@app.get("/fantasy/driver-profile")
def get_driver_profile(
    driver_code: str = Query(..., description="Driver code (e.g., 'VER', 'HAM')")
):
    """Get comprehensive driver performance profile"""

    results = vdb.search_similar_races(
        query=f"{driver_code} driver performance profile",
        top_k=1,
        filter_dict={'type': 'fantasy_driver_profile', 'driver_code': driver_code}
    )

    if not results:
        raise HTTPException(status_code=404, detail=f"Driver not found: {driver_code}")

    result = results[0]

    return {
        "driver_name": result['metadata']['driver_name'],
        "driver_code": result['metadata']['driver_code'],
        "team": result['metadata']['team'],
        "profile": result['text'],
        "relevance_score": round(result['score'], 3)
    }


@app.get("/fantasy/recent-form")
def get_drivers_by_form(
    momentum: str = Query("improving", description="'improving' or 'declining'"),
    limit: int = Query(10, ge=1, le=20)
):
    """Find drivers with good/bad recent form"""

    if momentum == "improving":
        query = "improving momentum recent form gaining positions good last 5 races"
    else:
        query = "declining momentum losing positions poor recent form"

    results = vdb.search_similar_races(
        query=query,
        top_k=limit,
        filter_dict={'type': 'fantasy_driver_profile'}
    )

    return {
        "momentum_filter": momentum,
        "drivers": [
            {
                "driver_name": r['metadata']['driver_name'],
                "driver_code": r['metadata']['driver_code'],
                "team": r['metadata']['team'],
                "relevance_score": round(r['score'], 3),
                "analysis": r['text']
            }
            for r in results
        ]
    }


@app.get("/fantasy/head-to-head")
def compare_drivers(
    driver1: str = Query(..., description="First driver code"),
    driver2: str = Query(..., description="Second driver code")
):
    """Compare two drivers head-to-head"""

    results = vdb.search_similar_races(
        query=f"{driver1} vs {driver2} head to head comparison",
        top_k=1,
        filter_dict={'type': 'fantasy_head_to_head'}
    )

    # Try reverse order
    if not results:
        results = vdb.search_similar_races(
            query=f"{driver2} vs {driver1} head to head comparison",
            top_k=1,
            filter_dict={'type': 'fantasy_head_to_head'}
        )

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No head-to-head data for {driver1} vs {driver2}"
        )

    return {
        "comparison": results[0]['text'],
        "driver1": results[0]['metadata']['driver1'],
        "driver2": results[0]['metadata']['driver2']
    }


@app.get("/fantasy/ask")
def natural_language_query(
    question: str = Query(..., description="Natural language question about F1 fantasy")
):
    """
    Ask a natural language question about F1 fantasy

    Examples:
    - "Who should I pick for Monaco?"
    - "Which drivers are in good form?"
    - "Is qualifying important at Monza?"
    """

    # Search across all fantasy documents
    results = vdb.search_similar_races(
        query=question,
        top_k=5
    )

    if not results:
        return {
            "question": question,
            "answer": "I don't have enough data to answer that question.",
            "sources": []
        }

    # Format context
    context_sources = []
    for i, r in enumerate(results, 1):
        context_sources.append({
            "rank": i,
            "relevance": round(r['score'], 3),
            "text": r['text'],
            "type": r['metadata'].get('type', 'unknown')
        })

    # Simple answer (in production, use LLM for better answers)
    answer = f"Based on the data, here are the top insights:\n\n"
    for source in context_sources[:3]:
        answer += f"• {source['text'][:200]}...\n\n"

    return {
        "question": question,
        "answer": answer,
        "sources": context_sources
    }


@app.get("/fantasy/circuits")
def list_circuits():
    """List all available circuits"""

    # Query for all circuit analyses
    results = vdb.search_similar_races(
        query="circuit analysis",
        top_k=50,
        filter_dict={'type': 'fantasy_circuit_analysis'}
    )

    circuits = list(set([r['metadata']['circuit'] for r in results]))

    return {
        "total": len(circuits),
        "circuits": sorted(circuits)
    }


@app.get("/fantasy/drivers")
def list_drivers():
    """List all available drivers"""

    # Query for all driver profiles
    results = vdb.search_similar_races(
        query="driver profile",
        top_k=100,
        filter_dict={'type': 'fantasy_driver_profile'}
    )

    drivers = {}
    for r in results:
        code = r['metadata']['driver_code']
        if code not in drivers:
            drivers[code] = {
                "code": code,
                "name": r['metadata']['driver_name'],
                "team": r['metadata']['team']
            }

    return {
        "total": len(drivers),
        "drivers": sorted(drivers.values(), key=lambda x: x['name'])
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
