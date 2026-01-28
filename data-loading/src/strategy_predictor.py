"""
F1 Race Strategy Predictor

Uses vector database to predict optimal race strategies based on:
- Historical race data from similar conditions
- Current session data (practice, qualifying, race)
- Real-time tire degradation
- Weather conditions
- Track characteristics

Provides strategy recommendations using RAG (Retrieval Augmented Generation)
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

import numpy as np
import pandas as pd

try:
    from .vector_db import F1VectorDB
except ImportError:
    from vector_db import F1VectorDB

logger = logging.getLogger(__name__)


@dataclass
class SessionData:
    """Current session data"""
    circuit: str
    session_type: str  # 'Practice', 'Qualifying', 'Race'
    lap_number: int
    total_laps: int

    # Conditions
    air_temp: float
    track_temp: float
    weather: str  # 'Dry', 'Wet', 'Mixed'

    # Tire data
    available_compounds: List[str]
    current_compound: Optional[str] = None
    tire_age: int = 0

    # Session info
    position: Optional[int] = None
    gap_to_leader: Optional[float] = None
    cars_ahead: Optional[int] = None


@dataclass
class StrategyRecommendation:
    """Strategy recommendation"""
    strategy_type: str  # '1-stop', '2-stop', '3-stop'
    stints: List[Dict[str, Any]]
    confidence: float
    reasoning: str
    similar_races: List[Dict[str, Any]]
    expected_position: Optional[int] = None


class F1StrategyPredictor:
    """Predicts optimal F1 race strategies"""

    def __init__(self, vector_db: Optional[F1VectorDB] = None):
        """
        Initialize strategy predictor

        Args:
            vector_db: F1 vector database instance
        """
        self.vector_db = vector_db or F1VectorDB()

        logger.info("Strategy Predictor initialized")

    def predict_optimal_strategy(
        self,
        session_data: SessionData,
        driver_position: Optional[int] = None,
        num_similar_races: int = 10
    ) -> StrategyRecommendation:
        """
        Predict optimal race strategy for current conditions

        Args:
            session_data: Current session data
            driver_position: Current driver position (if in race)
            num_similar_races: Number of similar races to analyze

        Returns:
            Strategy recommendation
        """
        logger.info(f"Predicting strategy for {session_data.circuit} ({session_data.weather})")

        # Step 1: Build context query
        context_query = self._build_context_query(session_data)

        # Step 2: Retrieve similar historical races
        similar_races = self.vector_db.search_similar_races(
            query=context_query,
            top_k=num_similar_races
        )

        # Step 3: Analyze winning strategies from similar races
        strategy_analysis = self._analyze_historical_strategies(similar_races, session_data)

        # Step 4: Generate recommendation
        recommendation = self._generate_recommendation(
            strategy_analysis,
            session_data,
            similar_races,
            driver_position
        )

        logger.info(f"Recommended: {recommendation.strategy_type} strategy")
        return recommendation

    def predict_pit_window(
        self,
        session_data: SessionData,
        current_stint_compound: str,
        stint_start_lap: int
    ) -> Tuple[int, int, str]:
        """
        Predict optimal pit window for current stint

        Args:
            session_data: Current session data
            current_stint_compound: Current tire compound
            stint_start_lap: Lap number when stint started

        Returns:
            Tuple of (earliest_lap, optimal_lap, latest_lap, reasoning)
        """
        # Query for similar tire stints
        query = (
            f"Circuit: {session_data.circuit} | "
            f"Compound: {current_stint_compound} | "
            f"Temperature: {session_data.track_temp}°C | "
            f"Weather: {session_data.weather}"
        )

        similar_strategies = self.vector_db.search_similar_strategies(
            query=query,
            circuit=session_data.circuit,
            top_k=20
        )

        # Analyze stint lengths
        stint_lengths = []

        for result in similar_strategies:
            try:
                metadata = result['metadata']
                data = json.loads(metadata.get('data', '{}'))

                if 'strategy' in data:
                    strategy = data['strategy']

                    for stint in strategy.get('stints', []):
                        if stint['compound'] == current_stint_compound:
                            stint_lengths.append(stint['stint_length'])

            except Exception:
                continue

        if not stint_lengths:
            # Default to conservative estimate
            default_lengths = {'SOFT': 20, 'MEDIUM': 30, 'HARD': 40}
            avg_stint = default_lengths.get(current_stint_compound.upper(), 25)
            logger.warning(f"No historical data, using default stint length: {avg_stint}")
        else:
            avg_stint = int(np.mean(stint_lengths))
            std_stint = int(np.std(stint_lengths))

            logger.info(f"Average {current_stint_compound} stint: {avg_stint} ± {std_stint} laps")

        # Calculate pit window
        current_lap = session_data.lap_number
        stint_age = current_lap - stint_start_lap

        earliest_lap = stint_start_lap + max(10, avg_stint - 10)
        optimal_lap = stint_start_lap + avg_stint
        latest_lap = stint_start_lap + min(session_data.total_laps - current_lap, avg_stint + 10)

        reasoning = (
            f"Based on {len(stint_lengths)} historical {current_stint_compound} stints at {session_data.circuit}. "
            f"Average stint length: {avg_stint} laps. "
            f"Current tire age: {stint_age} laps."
        )

        return earliest_lap, optimal_lap, latest_lap, reasoning

    def suggest_next_compound(
        self,
        session_data: SessionData,
        remaining_laps: int,
        used_compounds: List[str]
    ) -> Tuple[str, str]:
        """
        Suggest next tire compound

        Args:
            session_data: Current session data
            remaining_laps: Laps remaining in race
            used_compounds: Compounds already used (for mandatory 2-compound rule)

        Returns:
            Tuple of (recommended_compound, reasoning)
        """
        # Query similar races for successful strategies
        query = (
            f"Circuit: {session_data.circuit} | "
            f"Weather: {session_data.weather} | "
            f"Temperature: {session_data.track_temp}°C"
        )

        similar_strategies = self.vector_db.search_similar_strategies(
            query=query,
            circuit=session_data.circuit,
            top_k=15
        )

        # Analyze compound choices for remaining stint lengths
        compound_choices = {compound: 0 for compound in session_data.available_compounds}

        for result in similar_strategies:
            try:
                metadata = result['metadata']
                data = json.loads(metadata.get('data', '{}'))

                if 'strategy' in data:
                    strategy = data['strategy']

                    # Look at final stints with similar remaining laps
                    for i, stint in enumerate(strategy.get('stints', [])):
                        if abs(stint['stint_length'] - remaining_laps) < 10:
                            compound = stint['compound']
                            if compound in compound_choices:
                                compound_choices[compound] += 1

            except Exception:
                continue

        # Apply rules
        # Rule 1: Must use 2 different compounds in dry race
        if session_data.weather == 'Dry' and len(set(used_compounds)) < 2:
            # Must use a different compound
            unused = [c for c in session_data.available_compounds if c not in used_compounds]
            if unused:
                recommended = max(unused, key=lambda c: compound_choices.get(c, 0))
                reasoning = f"Mandatory 2-compound rule - must use {recommended}"
                return recommended, reasoning

        # Rule 2: Choose based on historical success
        if compound_choices:
            recommended = max(compound_choices, key=compound_choices.get)
            count = compound_choices[recommended]
            reasoning = (
                f"{recommended} compound used in {count} similar situations. "
                f"Estimated {remaining_laps} laps remaining."
            )
            return recommended, reasoning

        # Fallback: Conservative choice
        compound_hardness = {'SOFT': 1, 'MEDIUM': 2, 'HARD': 3}
        recommended = max(
            session_data.available_compounds,
            key=lambda c: compound_hardness.get(c.upper(), 2)
        )
        reasoning = "Conservative choice for remaining distance"

        return recommended, reasoning

    def _build_context_query(self, session_data: SessionData) -> str:
        """Build semantic search query from session data"""
        query_parts = [
            f"Circuit: {session_data.circuit}",
            f"Weather: {session_data.weather}",
            f"Track temperature: {session_data.track_temp}°C",
            f"Air temperature: {session_data.air_temp}°C"
        ]

        if session_data.available_compounds:
            compounds = ', '.join(session_data.available_compounds)
            query_parts.append(f"Tire compounds: {compounds}")

        return " | ".join(query_parts)

    def _analyze_historical_strategies(
        self,
        similar_races: List[Dict],
        session_data: SessionData
    ) -> Dict[str, Any]:
        """Analyze strategies from similar historical races"""
        strategy_counts = {'1-stop': 0, '2-stop': 0, '3-stop': 0}
        compound_sequences = []
        avg_pit_laps = []

        for race in similar_races:
            try:
                metadata = race['metadata']
                data_str = metadata.get('data', '{}')
                data = json.loads(data_str)

                # Extract strategy info
                if 'tire_strategies' in data:
                    for strategy in data['tire_strategies']:
                        stops = strategy.get('total_stops', 0)

                        if stops == 1:
                            strategy_counts['1-stop'] += 1
                        elif stops == 2:
                            strategy_counts['2-stop'] += 1
                        elif stops >= 3:
                            strategy_counts['3-stop'] += 1

                        # Collect compound sequences
                        stints = strategy.get('stints', [])
                        compounds = [stint['compound'] for stint in stints]
                        compound_sequences.append(compounds)

                        # Collect pit lap numbers
                        for stint in stints[:-1]:  # Exclude last stint
                            avg_pit_laps.append(stint['end_lap'])

            except Exception as e:
                logger.debug(f"Error parsing race data: {e}")
                continue

        # Calculate statistics
        total_strategies = sum(strategy_counts.values())

        if total_strategies > 0:
            strategy_percentages = {
                k: v / total_strategies for k, v in strategy_counts.items()
            }
        else:
            strategy_percentages = {'1-stop': 0.5, '2-stop': 0.4, '3-stop': 0.1}

        return {
            'strategy_distribution': strategy_percentages,
            'strategy_counts': strategy_counts,
            'compound_sequences': compound_sequences,
            'avg_pit_laps': avg_pit_laps,
            'total_samples': total_strategies
        }

    def _generate_recommendation(
        self,
        analysis: Dict[str, Any],
        session_data: SessionData,
        similar_races: List[Dict],
        driver_position: Optional[int]
    ) -> StrategyRecommendation:
        """Generate final strategy recommendation"""
        # Determine optimal strategy type
        strategy_dist = analysis['strategy_distribution']
        recommended_type = max(strategy_dist, key=strategy_dist.get)
        confidence = strategy_dist[recommended_type]

        # Generate stint plan
        num_stops = int(recommended_type.split('-')[0])
        stints = self._plan_stints(
            session_data,
            num_stops,
            analysis['compound_sequences']
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            recommended_type,
            analysis,
            session_data,
            confidence
        )

        return StrategyRecommendation(
            strategy_type=recommended_type,
            stints=stints,
            confidence=confidence,
            reasoning=reasoning,
            similar_races=similar_races[:5],  # Top 5 most similar
            expected_position=driver_position  # Placeholder
        )

    def _plan_stints(
        self,
        session_data: SessionData,
        num_stops: int,
        compound_sequences: List[List[str]]
    ) -> List[Dict[str, Any]]:
        """Plan stint structure"""
        total_laps = session_data.total_laps
        num_stints = num_stops + 1

        # Find most common compound sequence
        sequence_counts = {}
        for seq in compound_sequences:
            if len(seq) == num_stints:
                seq_tuple = tuple(seq)
                sequence_counts[seq_tuple] = sequence_counts.get(seq_tuple, 0) + 1

        if sequence_counts:
            best_sequence = max(sequence_counts, key=sequence_counts.get)
        else:
            # Default sequence
            if num_stints == 2:
                best_sequence = ('MEDIUM', 'HARD')
            elif num_stints == 3:
                best_sequence = ('SOFT', 'MEDIUM', 'HARD')
            else:
                best_sequence = ('MEDIUM',) * num_stints

        # Estimate stint lengths (distribute laps)
        base_stint_length = total_laps // num_stints
        stints = []

        current_lap = 1
        for i, compound in enumerate(best_sequence):
            if i == num_stints - 1:
                # Last stint takes remaining laps
                stint_length = total_laps - current_lap + 1
            else:
                stint_length = base_stint_length

            stints.append({
                'stint_number': i + 1,
                'compound': compound,
                'start_lap': current_lap,
                'end_lap': current_lap + stint_length - 1,
                'stint_length': stint_length,
                'pit_lap': current_lap + stint_length if i < num_stints - 1 else None
            })

            current_lap += stint_length

        return stints

    def _generate_reasoning(
        self,
        strategy_type: str,
        analysis: Dict[str, Any],
        session_data: SessionData,
        confidence: float
    ) -> str:
        """Generate human-readable reasoning"""
        parts = []

        parts.append(
            f"Recommended {strategy_type} strategy based on analysis of "
            f"{analysis['total_samples']} similar races at {session_data.circuit}."
        )

        parts.append(
            f"Strategy distribution: "
            f"1-stop ({analysis['strategy_distribution']['1-stop']:.1%}), "
            f"2-stop ({analysis['strategy_distribution']['2-stop']:.1%}), "
            f"3-stop ({analysis['strategy_distribution']['3-stop']:.1%})."
        )

        parts.append(
            f"Current conditions: {session_data.weather} weather, "
            f"{session_data.track_temp}°C track temperature."
        )

        parts.append(f"Confidence level: {confidence:.1%}")

        return " ".join(parts)

    def get_live_strategy_update(
        self,
        session_data: SessionData,
        original_plan: StrategyRecommendation,
        actual_lap: int
    ) -> Dict[str, Any]:
        """
        Provide live strategy updates during race

        Args:
            session_data: Current session data
            original_plan: Original strategy recommendation
            actual_lap: Current race lap

        Returns:
            Strategy update with adjustments
        """
        # Find current stint in plan
        current_stint = None
        for stint in original_plan.stints:
            if stint['start_lap'] <= actual_lap <= stint['end_lap']:
                current_stint = stint
                break

        if not current_stint:
            return {'status': 'error', 'message': 'Current lap outside planned stints'}

        # Check if on-plan
        laps_until_pit = current_stint['end_lap'] - actual_lap
        on_plan = abs(laps_until_pit) <= 3  # Within 3 laps is "on plan"

        # Generate update
        update = {
            'current_lap': actual_lap,
            'current_stint': current_stint['stint_number'],
            'current_compound': current_stint['compound'],
            'laps_until_planned_pit': laps_until_pit,
            'on_plan': on_plan,
            'recommendation': 'Stay out' if on_plan else 'Consider adjusting'
        }

        return update


if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    load_dotenv()

    # Initialize predictor
    predictor = F1StrategyPredictor()

    # Example session data
    session = SessionData(
        circuit="Monza",
        session_type="Race",
        lap_number=1,
        total_laps=53,
        air_temp=25.0,
        track_temp=35.0,
        weather="Dry",
        available_compounds=["SOFT", "MEDIUM", "HARD"]
    )

    # Get strategy recommendation
    recommendation = predictor.predict_optimal_strategy(session)

    print("\n=== STRATEGY RECOMMENDATION ===")
    print(f"Strategy: {recommendation.strategy_type}")
    print(f"Confidence: {recommendation.confidence:.1%}")
    print(f"\nReasoning: {recommendation.reasoning}")
    print("\nStint Plan:")
    for stint in recommendation.stints:
        print(f"  Stint {stint['stint_number']}: {stint['compound']} "
              f"(Laps {stint['start_lap']}-{stint['end_lap']}, {stint['stint_length']} laps)")
