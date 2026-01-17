"""
F1 FastAPI Backend

REST API server for the F1 Race Strategy Analyzer.
Provides endpoints for:
- Race data retrieval
- Strategy generation
- ML predictions
- Semantic search
- What-if analysis

Author: F1 Race Strategy Analyzer
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

class RaceInfo(BaseModel):
    """Race information for strategy generation"""
    season: int = Field(default=2024, description="Race season year")
    round: Optional[int] = Field(default=None, description="Race round number")
    circuit: str = Field(..., description="Circuit identifier or name")
    weather_forecast: str = Field(default="Unknown", description="Weather conditions")
    total_laps: int = Field(default=50, description="Total race laps")
    driver: Optional[str] = Field(default=None, description="Driver name")
    constructor: Optional[str] = Field(default=None, description="Constructor/team name")
    grid_position: Optional[int] = Field(default=None, ge=1, le=20, description="Starting position")
    tire_compounds: List[str] = Field(
        default=["SOFT", "MEDIUM", "HARD"],
        description="Available tire compounds"
    )


class StrategyRequest(BaseModel):
    """Request for strategy generation"""
    race_info: RaceInfo
    include_historical: bool = Field(default=True, description="Include historical context")
    include_predictions: bool = Field(default=True, description="Include ML predictions")


class StrategyResponse(BaseModel):
    """Generated strategy response"""
    executive_summary: str
    recommended_strategy: Dict[str, Any]
    alternative_strategies: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    weather_contingency: str
    key_decision_points: List[Dict[str, Any]]
    confidence_score: float
    generated_at: str


class QueryRequest(BaseModel):
    """Semantic query request"""
    query: str = Field(..., description="Natural language query")
    circuit: Optional[str] = Field(default=None, description="Filter by circuit")
    season: Optional[int] = Field(default=None, description="Filter by season")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")


class WhatIfRequest(BaseModel):
    """What-if scenario request"""
    circuit: str
    current_lap: int
    total_laps: int
    position: int
    current_tire: str
    tire_age: int
    pit_stops_done: int = 0
    scenario_description: str
    question: str


class PredictionRequest(BaseModel):
    """ML prediction request"""
    race_id: Optional[str] = None
    driver_id: Optional[str] = None
    current_lap: int = Field(default=1, ge=1)
    current_position: int = Field(default=10, ge=1, le=20)
    tire_compound: str = Field(default="MEDIUM")
    tire_age: int = Field(default=0, ge=0)


class DataIngestionRequest(BaseModel):
    """Data ingestion request"""
    season: int
    round: Optional[int] = None
    force_refresh: bool = False


# =============================================================================
# Application State
# =============================================================================

class AppState:
    """Application state container"""
    
    def __init__(self):
        self.data_collector = None
        self.vector_db = None
        self.strategy_generator = None
        self.ml_models = None
        self.feature_engineer = None
        
    def initialize(self):
        """Initialize all components"""
        logger.info("Initializing application components...")
        
        try:
            from data_collection import F1DataCollector
            self.data_collector = F1DataCollector(use_fastf1=False)
            logger.info("Data collector initialized")
        except Exception as e:
            logger.warning(f"Data collector init failed: {e}")
        
        try:
            from vector_database import F1VectorDatabase
            self.vector_db = F1VectorDatabase()
            self.vector_db.create_index()
            logger.info("Vector database initialized")
        except Exception as e:
            logger.warning(f"Vector database init failed: {e}")
        
        try:
            from llm_strategy_generator import F1StrategyGenerator
            self.strategy_generator = F1StrategyGenerator()
            logger.info("Strategy generator initialized")
        except Exception as e:
            logger.warning(f"Strategy generator init failed: {e}")
        
        try:
            from feature_engineering import F1FeatureEngineer
            self.feature_engineer = F1FeatureEngineer()
            logger.info("Feature engineer initialized")
        except Exception as e:
            logger.warning(f"Feature engineer init failed: {e}")
        
        try:
            from tensorflow_models import EnsemblePredictor
            self.ml_models = EnsemblePredictor()
            # Try to load pre-trained models
            try:
                self.ml_models.load_all_models('./models')
                logger.info("ML models loaded")
            except:
                logger.info("No pre-trained models found")
        except Exception as e:
            logger.warning(f"ML models init failed: {e}")
        
        logger.info("Application initialization complete")


# Global state
app_state = AppState()


# =============================================================================
# FastAPI Application
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting F1 Race Strategy Analyzer API...")
    app_state.initialize()
    yield
    # Shutdown
    logger.info("Shutting down F1 Race Strategy Analyzer API...")


app = FastAPI(
    title="F1 Race Strategy Analyzer API",
    description="""
    AI-powered Formula 1 race strategy analysis system.
    
    Features:
    - Race data retrieval from multiple sources
    - AI-generated race strategies using Claude
    - ML predictions for race outcomes and pit stops
    - Semantic search over historical strategies
    - What-if scenario analysis
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health & Info Endpoints
# =============================================================================

@app.get("/", tags=["Info"])
async def root():
    """API root endpoint"""
    return {
        "name": "F1 Race Strategy Analyzer API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", tags=["Info"])
async def health_check():
    """Health check endpoint"""
    components = {
        "data_collector": app_state.data_collector is not None,
        "vector_database": app_state.vector_db is not None,
        "strategy_generator": app_state.strategy_generator is not None,
        "ml_models": app_state.ml_models is not None
    }
    
    return {
        "status": "healthy" if any(components.values()) else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "components": components
    }


# =============================================================================
# Race Data Endpoints
# =============================================================================

@app.get("/races", tags=["Race Data"])
async def list_races(
    season: int = Query(default=2024, description="Season year"),
    limit: int = Query(default=20, ge=1, le=50)
):
    """List races for a season"""
    if not app_state.data_collector:
        raise HTTPException(503, "Data collector not available")
    
    try:
        schedule = app_state.data_collector.ergast.get_season_schedule(season)
        return {
            "season": season,
            "races": schedule[:limit],
            "total": len(schedule)
        }
    except Exception as e:
        logger.error(f"Failed to list races: {e}")
        raise HTTPException(500, str(e))


@app.get("/race/{season}/{round_num}", tags=["Race Data"])
async def get_race(
    season: int,
    round_num: int,
    include_lap_times: bool = Query(default=False),
    include_pit_stops: bool = Query(default=True)
):
    """Get detailed race data"""
    if not app_state.data_collector:
        raise HTTPException(503, "Data collector not available")
    
    try:
        # Get race results
        results = app_state.data_collector.ergast.get_race_results(season, round_num)
        schedule = app_state.data_collector.ergast.get_season_schedule(season)
        race_info = next((r for r in schedule if int(r['round']) == round_num), None)
        
        if not race_info:
            raise HTTPException(404, f"Race not found: {season} Round {round_num}")
        
        response = {
            "season": season,
            "round": round_num,
            "race_name": race_info.get('raceName', ''),
            "circuit": race_info.get('Circuit', {}),
            "date": race_info.get('date', ''),
            "results": results
        }
        
        if include_pit_stops:
            pit_stops = app_state.data_collector.ergast.get_pit_stops(season, round_num)
            response["pit_stops"] = pit_stops
        
        if include_lap_times:
            lap_times = app_state.data_collector.ergast.get_lap_times(season, round_num)
            response["lap_times"] = lap_times[:10]  # First 10 laps
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get race data: {e}")
        raise HTTPException(500, str(e))


@app.get("/race/{race_id}/predictions", tags=["Race Data"])
async def get_race_predictions(race_id: str):
    """Get ML predictions for a race"""
    # Parse race_id (format: "2024_monaco_gp" or "2024_5")
    parts = race_id.split('_')
    
    if not app_state.ml_models:
        # Return placeholder predictions
        return {
            "race_id": race_id,
            "predictions": {
                "note": "ML models not loaded. Showing placeholder predictions.",
                "predicted_winner": "verstappen",
                "podium_probability": 0.85,
                "safety_car_probability": 0.35
            }
        }
    
    try:
        # In production, this would use actual ML models
        return {
            "race_id": race_id,
            "predictions": {
                "position_predictions": [],
                "pit_stop_windows": [],
                "tire_strategy": {}
            }
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/data/ingest", tags=["Race Data"])
async def ingest_race_data(
    request: DataIngestionRequest,
    background_tasks: BackgroundTasks
):
    """Trigger data ingestion for a race or season"""
    if not app_state.data_collector:
        raise HTTPException(503, "Data collector not available")
    
    async def ingest_task():
        try:
            if request.round:
                data = app_state.data_collector.collect_race_data(
                    request.season, request.round
                )
                logger.info(f"Ingested: {request.season} R{request.round}")
            else:
                data = app_state.data_collector.get_seasons_data(
                    request.season, request.season
                )
                logger.info(f"Ingested season: {request.season}")
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
    
    background_tasks.add_task(ingest_task)
    
    return {
        "status": "accepted",
        "message": f"Data ingestion started for {request.season}" + 
                   (f" Round {request.round}" if request.round else "")
    }


# =============================================================================
# Strategy Generation Endpoints
# =============================================================================

@app.post("/strategy/generate", response_model=StrategyResponse, tags=["Strategy"])
async def generate_strategy(request: StrategyRequest):
    """Generate a comprehensive race strategy"""
    if not app_state.strategy_generator:
        raise HTTPException(503, "Strategy generator not available")
    
    try:
        # Get historical context
        historical_context = []
        if request.include_historical and app_state.vector_db:
            query = f"Race strategy for {request.race_info.circuit}"
            historical_context = app_state.vector_db.search_similar_strategies(
                query, top_k=5
            )
        
        # Get ML predictions
        ml_predictions = None
        if request.include_predictions and app_state.ml_models:
            # Would use actual ML models here
            pass
        
        # Generate strategy
        race_info = {
            'circuit': request.race_info.circuit,
            'weather_forecast': request.race_info.weather_forecast,
            'total_laps': request.race_info.total_laps,
            'driver': request.race_info.driver,
            'constructor': request.race_info.constructor,
            'grid_position': request.race_info.grid_position,
            'tire_compounds': request.race_info.tire_compounds
        }
        
        strategy = app_state.strategy_generator.generate_race_strategy(
            race_info=race_info,
            historical_context=historical_context,
            ml_predictions=ml_predictions
        )
        
        return StrategyResponse(
            executive_summary=strategy.executive_summary,
            recommended_strategy=strategy.recommended_strategy,
            alternative_strategies=strategy.alternative_strategies,
            risk_assessment=strategy.risk_assessment,
            weather_contingency=strategy.weather_contingency,
            key_decision_points=strategy.key_decision_points,
            confidence_score=strategy.confidence_score,
            generated_at=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Strategy generation failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/strategy/compare", tags=["Strategy"])
async def compare_strategies(
    strategy_a: Dict[str, Any],
    strategy_b: Dict[str, Any],
    race_info: RaceInfo
):
    """Compare two race strategies"""
    if not app_state.strategy_generator:
        raise HTTPException(503, "Strategy generator not available")
    
    try:
        comparison = app_state.strategy_generator.compare_strategies(
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            race_context={
                'circuit': race_info.circuit,
                'total_laps': race_info.total_laps,
                'weather': race_info.weather_forecast
            }
        )
        return comparison
    except Exception as e:
        logger.error(f"Strategy comparison failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/strategy/what-if", tags=["Strategy"])
async def analyze_what_if(request: WhatIfRequest):
    """Analyze a what-if scenario"""
    if not app_state.strategy_generator:
        raise HTTPException(503, "Strategy generator not available")
    
    try:
        base_scenario = {
            'circuit': request.circuit,
            'current_lap': request.current_lap,
            'total_laps': request.total_laps,
            'position': request.position,
            'current_tire': request.current_tire,
            'tire_age': request.tire_age,
            'pit_stops_done': request.pit_stops_done
        }
        
        analysis = app_state.strategy_generator.analyze_what_if(
            base_scenario=base_scenario,
            what_if_description=request.scenario_description,
            question=request.question
        )
        
        return analysis
        
    except Exception as e:
        logger.error(f"What-if analysis failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/strategy/quick-recommendation", tags=["Strategy"])
async def quick_pit_recommendation(
    circuit: str,
    current_lap: int,
    total_laps: int,
    current_tire: str,
    tire_age: int,
    position: int
):
    """Get a quick pit stop recommendation"""
    if not app_state.strategy_generator:
        # Provide basic recommendation without LLM
        should_pit = tire_age > 25 or (tire_age > 15 and total_laps - current_lap < 20)
        return {
            "pit_now": should_pit,
            "recommended_compound": "HARD" if should_pit and total_laps - current_lap > 15 else "SOFT",
            "reasoning": "Basic recommendation (LLM unavailable)"
        }
    
    from llm_strategy_generator import StrategyAssistant
    
    assistant = StrategyAssistant(
        vector_db=app_state.vector_db,
        ml_models=app_state.ml_models
    )
    
    return assistant.quick_recommendation(
        circuit=circuit,
        current_lap=current_lap,
        total_laps=total_laps,
        current_tire=current_tire,
        tire_age=tire_age,
        position=position
    )


# =============================================================================
# Semantic Search Endpoints
# =============================================================================

@app.post("/query", tags=["Search"])
async def semantic_query(request: QueryRequest):
    """Semantic search over historical strategies"""
    if not app_state.vector_db:
        raise HTTPException(503, "Vector database not available")
    
    try:
        filter_dict = {}
        if request.circuit:
            filter_dict['circuit_id'] = request.circuit.lower()
        if request.season:
            filter_dict['season'] = request.season
        
        results = app_state.vector_db.search_similar_strategies(
            query=request.query,
            top_k=request.top_k,
            filter_dict=filter_dict if filter_dict else None
        )
        
        return {
            "query": request.query,
            "filters": filter_dict,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Semantic query failed: {e}")
        raise HTTPException(500, str(e))


@app.get("/search/circuit/{circuit_id}", tags=["Search"])
async def search_by_circuit(
    circuit_id: str,
    query: Optional[str] = None,
    top_k: int = Query(default=10, ge=1, le=50)
):
    """Search strategies for a specific circuit"""
    if not app_state.vector_db:
        raise HTTPException(503, "Vector database not available")
    
    try:
        results = app_state.vector_db.search_by_circuit(
            circuit_id=circuit_id.lower(),
            query=query,
            top_k=top_k
        )
        
        return {
            "circuit": circuit_id,
            "query": query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Circuit search failed: {e}")
        raise HTTPException(500, str(e))


@app.get("/search/winning-strategies", tags=["Search"])
async def get_winning_strategies(
    circuit_id: Optional[str] = None,
    top_k: int = Query(default=10, ge=1, le=50)
):
    """Get strategies that resulted in race wins"""
    if not app_state.vector_db:
        raise HTTPException(503, "Vector database not available")
    
    try:
        results = app_state.vector_db.get_winning_strategies(
            circuit_id=circuit_id.lower() if circuit_id else None,
            top_k=top_k
        )
        
        return {
            "circuit": circuit_id,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Winning strategies search failed: {e}")
        raise HTTPException(500, str(e))


# =============================================================================
# ML Prediction Endpoints
# =============================================================================

@app.post("/predict/position", tags=["Predictions"])
async def predict_position(request: PredictionRequest):
    """Predict race finishing position"""
    if not app_state.ml_models or not app_state.ml_models.race_predictor:
        # Return placeholder prediction
        return {
            "predicted_position": 5,
            "confidence": 0.65,
            "note": "ML model not available, showing placeholder"
        }
    
    try:
        import numpy as np
        
        # In production, would prepare proper features
        sample_sequence = np.random.randn(1, 10, 30)
        position, probs = app_state.ml_models.race_predictor.predict_position(sample_sequence)
        
        return {
            "predicted_position": position,
            "position_probabilities": probs[:5].tolist(),
            "top_3_probability": float(probs[:3].sum())
        }
    except Exception as e:
        logger.error(f"Position prediction failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/predict/pit-stop", tags=["Predictions"])
async def predict_pit_stop(request: PredictionRequest):
    """Get pit stop recommendation from ML model"""
    if not app_state.ml_models or not app_state.ml_models.pit_optimizer:
        # Basic recommendation
        should_pit = request.tire_age > 25
        return {
            "recommended_action": "Pit - HARD" if should_pit else "Stay out",
            "confidence": 0.7,
            "note": "ML model not available, using heuristic"
        }
    
    try:
        import numpy as np
        
        # Prepare state features
        state = np.array([
            request.current_lap / 50,
            request.current_position / 20,
            request.tire_age / 30,
            0.5, 0.5, 0.5, 0.5,  # Placeholder features
            0.5, 0.5, 0.5, 0.5,
            0.5, 0.5, 0.5, 0.5
        ])
        
        strategy = app_state.ml_models.pit_optimizer.get_optimal_strategy(
            state, remaining_laps=50 - request.current_lap
        )
        
        return strategy
        
    except Exception as e:
        logger.error(f"Pit stop prediction failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/predict/tire-degradation", tags=["Predictions"])
async def predict_tire_degradation(
    compound: str,
    stint_length: int = Query(default=20, ge=1, le=50),
    track_temp: float = Query(default=30.0),
    fuel_load: float = Query(default=100.0)
):
    """Predict tire degradation for a stint"""
    if not app_state.ml_models or not app_state.ml_models.degradation_model:
        # Return estimated degradation
        base_deg = {'SOFT': 0.15, 'MEDIUM': 0.10, 'HARD': 0.06}.get(compound.upper(), 0.10)
        
        return {
            "compound": compound.upper(),
            "stint_length": stint_length,
            "predicted_degradation_per_lap": base_deg,
            "total_time_loss": base_deg * stint_length,
            "note": "ML model not available, using estimates"
        }
    
    try:
        import numpy as np
        
        features = np.array([
            0, fuel_load / 110, track_temp / 60, 25 / 40,
            0.7, 0.5, 200 / 250, 0.5
        ])
        
        result = app_state.ml_models.degradation_model.predict_stint(features, stint_length)
        
        return {
            "compound": compound,
            "stint_length": stint_length,
            "lap_times": result['lap_times'].tolist(),
            "total_time": float(result['total_time']),
            "avg_degradation": float(result['avg_degradation'])
        }
        
    except Exception as e:
        logger.error(f"Tire degradation prediction failed: {e}")
        raise HTTPException(500, str(e))


# =============================================================================
# Vector Database Management Endpoints
# =============================================================================

@app.get("/vector-db/stats", tags=["Admin"])
async def get_vector_db_stats():
    """Get vector database statistics"""
    if not app_state.vector_db:
        raise HTTPException(503, "Vector database not available")
    
    return app_state.vector_db.get_index_stats()


@app.post("/vector-db/index-strategies", tags=["Admin"])
async def index_strategies(
    season: int,
    background_tasks: BackgroundTasks
):
    """Index race strategies from a season into the vector database"""
    if not app_state.vector_db or not app_state.data_collector:
        raise HTTPException(503, "Required components not available")
    
    async def index_task():
        try:
            from vector_database import StrategyIndexer
            
            # Collect season data
            data = app_state.data_collector.get_seasons_data(season, season)
            
            if data and data[0].get('races'):
                indexer = StrategyIndexer(app_state.vector_db)
                count = indexer.index_race_strategies(data[0]['races'])
                logger.info(f"Indexed {count} strategies from {season}")
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
    
    background_tasks.add_task(index_task)
    
    return {
        "status": "accepted",
        "message": f"Indexing started for season {season}"
    }


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url)
        }
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the FastAPI server"""
    port = int(os.environ.get('PORT', 8000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"Starting F1 Race Strategy Analyzer API on {host}:{port}")
    
    uvicorn.run(
        "fastapi_backend:app",
        host=host,
        port=port,
        reload=os.environ.get('DEBUG', 'false').lower() == 'true',
        log_level="info"
    )


if __name__ == "__main__":
    main()
