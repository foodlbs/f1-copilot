"""
F1 Race Knowledge Base & Strategy Predictor

A comprehensive system for F1 race analysis using vector databases
and historical race data.
"""

__version__ = "1.0.0"

from .data_collector import F1DataCollector, RaceWeekendData
from .vector_db import F1VectorDB, VectorDBConfig, EmbeddingGenerator
from .knowledge_base_builder import F1KnowledgeBaseBuilder
from .strategy_predictor import F1StrategyPredictor, SessionData, StrategyRecommendation
from .race_simulator import RaceSimulator, Driver, RaceState

__all__ = [
    "F1DataCollector",
    "RaceWeekendData",
    "F1VectorDB",
    "VectorDBConfig",
    "EmbeddingGenerator",
    "F1KnowledgeBaseBuilder",
    "F1StrategyPredictor",
    "SessionData",
    "StrategyRecommendation",
    "RaceSimulator",
    "Driver",
    "RaceState",
]
