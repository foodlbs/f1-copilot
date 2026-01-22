"""
F1 Fantasy-Optimized Data Ingestion

This module extends the CSV ingestion to include fantasy-relevant metrics:
- Driver performance trends and consistency
- Circuit-specific historical performance
- Head-to-head driver comparisons
- Constructor reliability and performance
- Qualifying-to-race position changes
- Points per million (value picks)
- Recent form and momentum

Perfect for powering F1 fantasy lineup recommendation systems.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import json

from dotenv import load_dotenv

try:
    from .csv_data_ingestion import F1CSVDataLoader
    from .vector_db import F1VectorDB
except ImportError:
    from csv_data_ingestion import F1CSVDataLoader
    from vector_db import F1VectorDB

# Configure logging
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'fantasy_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class F1FantasyDataProcessor:
    """Process F1 data with fantasy league metrics"""

    def __init__(self, loader: F1CSVDataLoader):
        self.loader = loader

    def calculate_driver_performance_metrics(
        self,
        race_data: List[Dict[str, Any]],
        driver_code: str
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics for a driver"""

        driver_races = []
        for race in race_data:
            for result in race.get('race_results', []):
                if result.get('driver_code') == driver_code:
                    driver_races.append({
                        'race': race['race_name'],
                        'season': race['season'],
                        'circuit': race['circuit_name'],
                        'position': result.get('position'),
                        'grid': result.get('grid_position'),
                        'points': result.get('points', 0),
                        'status': result.get('status'),
                        'team': result.get('team')
                    })

        if not driver_races:
            return {}

        # Calculate metrics
        positions = [r['position'] for r in driver_races if r['position']]
        points_scored = [r['points'] for r in driver_races]
        grid_positions = [r['grid'] for r in driver_races if r['grid']]

        metrics = {
            'total_races': len(driver_races),
            'avg_finish_position': np.mean(positions) if positions else None,
            'best_finish': min(positions) if positions else None,
            'worst_finish': max(positions) if positions else None,
            'avg_points_per_race': np.mean(points_scored) if points_scored else 0,
            'total_points': sum(points_scored),
            'podiums': len([p for p in positions if p and p <= 3]),
            'top_10s': len([p for p in positions if p and p <= 10]),
            'dnf_rate': len([r for r in driver_races if r['status'] not in ['Finished', '+1 Lap', '+2 Laps']]) / len(driver_races) if driver_races else 0,
            'avg_grid_position': np.mean(grid_positions) if grid_positions else None,
        }

        # Position changes (overtakes/places lost)
        position_changes = []
        for r in driver_races:
            if r['position'] and r['grid']:
                change = r['grid'] - r['position']  # Positive = gained positions
                position_changes.append(change)

        metrics['avg_position_change'] = np.mean(position_changes) if position_changes else 0
        metrics['positions_gained_total'] = sum([c for c in position_changes if c > 0])

        # Consistency (standard deviation of finishing positions)
        metrics['consistency_score'] = 1 / (np.std(positions) + 1) if len(positions) > 1 else 0

        return metrics

    def calculate_circuit_specific_performance(
        self,
        race_data: List[Dict[str, Any]],
        driver_code: str,
        circuit_name: str
    ) -> Dict[str, Any]:
        """Calculate driver's historical performance at a specific circuit"""

        circuit_races = []
        for race in race_data:
            if race['circuit_name'] == circuit_name:
                for result in race.get('race_results', []):
                    if result.get('driver_code') == driver_code:
                        circuit_races.append({
                            'season': race['season'],
                            'position': result.get('position'),
                            'points': result.get('points', 0),
                            'grid': result.get('grid_position')
                        })

        if not circuit_races:
            return {}

        positions = [r['position'] for r in circuit_races if r['position']]

        return {
            'races_at_circuit': len(circuit_races),
            'avg_finish': np.mean(positions) if positions else None,
            'best_finish': min(positions) if positions else None,
            'avg_points': np.mean([r['points'] for r in circuit_races]),
            'total_points': sum([r['points'] for r in circuit_races]),
            'last_3_avg': np.mean(positions[-3:]) if len(positions) >= 3 else None
        }

    def calculate_recent_form(
        self,
        race_data: List[Dict[str, Any]],
        driver_code: str,
        last_n_races: int = 5
    ) -> Dict[str, Any]:
        """Calculate driver's recent form (momentum)"""

        # Sort by season and round
        sorted_races = sorted(race_data, key=lambda x: (x['season'], x['round']))

        recent_results = []
        for race in sorted_races[-last_n_races:]:
            for result in race.get('race_results', []):
                if result.get('driver_code') == driver_code:
                    recent_results.append({
                        'position': result.get('position'),
                        'points': result.get('points', 0)
                    })

        if not recent_results:
            return {}

        positions = [r['position'] for r in recent_results if r['position']]
        points = [r['points'] for r in recent_results]

        # Calculate momentum (improving = negative trend in positions)
        if len(positions) >= 2:
            # Linear regression on positions (negative slope = improving)
            x = np.arange(len(positions))
            slope, _ = np.polyfit(x, positions, 1)
            momentum = -slope  # Positive momentum = improving positions
        else:
            momentum = 0

        return {
            'last_n_races': last_n_races,
            'avg_finish': np.mean(positions) if positions else None,
            'avg_points': np.mean(points) if points else 0,
            'momentum_score': momentum,
            'improving': momentum > 0.5,  # Gaining more than 0.5 positions per race
            'recent_positions': positions
        }

    def calculate_constructor_reliability(
        self,
        race_data: List[Dict[str, Any]],
        constructor_name: str
    ) -> Dict[str, Any]:
        """Calculate constructor reliability and performance metrics"""

        constructor_results = []
        for race in race_data:
            race_points = 0
            dnfs = 0
            finishes = 0

            for result in race.get('race_results', []):
                if result.get('team') == constructor_name:
                    race_points += result.get('points', 0)
                    if result.get('status') not in ['Finished', '+1 Lap', '+2 Laps']:
                        dnfs += 1
                    else:
                        finishes += 1

            if dnfs + finishes > 0:  # Constructor participated in this race
                constructor_results.append({
                    'race': race['race_name'],
                    'season': race['season'],
                    'points': race_points,
                    'dnfs': dnfs,
                    'finishes': finishes
                })

        if not constructor_results:
            return {}

        return {
            'total_races': len(constructor_results),
            'avg_points_per_race': np.mean([r['points'] for r in constructor_results]),
            'total_points': sum([r['points'] for r in constructor_results]),
            'reliability_rate': sum([r['finishes'] for r in constructor_results]) / (sum([r['finishes'] + r['dnfs'] for r in constructor_results])),
            'total_dnfs': sum([r['dnfs'] for r in constructor_results])
        }

    def generate_fantasy_driver_document(
        self,
        race_data: List[Dict[str, Any]],
        driver_code: str,
        driver_name: str,
        team: str,
        upcoming_circuit: Optional[str] = None
    ) -> str:
        """Generate fantasy-optimized document for a driver"""

        parts = []

        # Driver identity
        parts.append(f"Driver: {driver_name} ({driver_code})")
        parts.append(f"Team: {team}")

        # Overall performance
        overall = self.calculate_driver_performance_metrics(race_data, driver_code)
        if overall:
            parts.append(f"Season avg finish: P{overall['avg_finish_position']:.1f}")
            parts.append(f"Avg points per race: {overall['avg_points_per_race']:.1f}")
            parts.append(f"Podiums: {overall['podiums']}, Top 10s: {overall['top_10s']}")
            parts.append(f"Position changes avg: {overall['avg_position_change']:+.1f}")
            parts.append(f"DNF rate: {overall['dnf_rate']:.1%}")
            parts.append(f"Consistency: {overall['consistency_score']:.2f}")

        # Recent form
        recent_form = self.calculate_recent_form(race_data, driver_code, last_n_races=5)
        if recent_form:
            momentum_desc = "improving" if recent_form['improving'] else "declining"
            parts.append(f"Recent form (last 5): Avg P{recent_form['avg_finish']:.1f}, {recent_form['avg_points']:.1f} pts/race")
            parts.append(f"Momentum: {momentum_desc} (score: {recent_form['momentum_score']:+.2f})")

        # Circuit-specific performance (if upcoming circuit provided)
        if upcoming_circuit:
            circuit_perf = self.calculate_circuit_specific_performance(race_data, driver_code, upcoming_circuit)
            if circuit_perf:
                parts.append(f"At {upcoming_circuit}: {circuit_perf['races_at_circuit']} races, avg P{circuit_perf['avg_finish']:.1f}, best P{circuit_perf['best_finish']}")

        return " | ".join(parts)

    def generate_fantasy_circuit_document(
        self,
        race_data: List[Dict[str, Any]],
        circuit_name: str
    ) -> str:
        """Generate circuit analysis for fantasy recommendations"""

        circuit_races = [r for r in race_data if r['circuit_name'] == circuit_name]

        if not circuit_races:
            return f"Circuit: {circuit_name} | No historical data"

        parts = []
        parts.append(f"Circuit: {circuit_name}")
        parts.append(f"Historical races: {len(circuit_races)}")

        # Overtaking analysis (avg position changes)
        avg_changes = []
        for race in circuit_races:
            for result in race.get('race_results', []):
                if result.get('position') and result.get('grid_position'):
                    change = abs(result['grid_position'] - result['position'])
                    avg_changes.append(change)

        if avg_changes:
            avg_overtaking = np.mean(avg_changes)
            overtaking_type = "high overtaking" if avg_overtaking > 3 else "low overtaking" if avg_overtaking < 1.5 else "moderate overtaking"
            parts.append(f"Overtaking: {overtaking_type} (avg {avg_overtaking:.1f} position changes)")

        # Strategy analysis (avg pit stops)
        avg_stops = []
        for race in circuit_races:
            for strategy in race.get('tire_strategies', []):
                if strategy.get('total_stops'):
                    avg_stops.append(strategy['total_stops'])

        if avg_stops:
            parts.append(f"Typical strategy: {np.mean(avg_stops):.1f} stop(s)")

        # Qualifying importance
        quali_advantage = []
        for race in circuit_races:
            for result in race.get('race_results', []):
                if result.get('position') and result.get('grid_position'):
                    if result['grid_position'] <= 3:  # Top 3 quali
                        quali_advantage.append(result['position'])

        if quali_advantage:
            avg_finish_from_top3_quali = np.mean(quali_advantage)
            quali_importance = "very high" if avg_finish_from_top3_quali < 3.5 else "high" if avg_finish_from_top3_quali < 5 else "moderate"
            parts.append(f"Qualifying importance: {quali_importance} (top 3 quali avg finish P{avg_finish_from_top3_quali:.1f})")

        # DNF rate at circuit
        dnfs = 0
        total_results = 0
        for race in circuit_races:
            for result in race.get('race_results', []):
                total_results += 1
                if result.get('status') not in ['Finished', '+1 Lap', '+2 Laps']:
                    dnfs += 1

        if total_results > 0:
            dnf_rate = dnfs / total_results
            reliability_desc = "high risk" if dnf_rate > 0.15 else "moderate risk" if dnf_rate > 0.08 else "low risk"
            parts.append(f"Circuit reliability: {reliability_desc} ({dnf_rate:.1%} DNF rate)")

        return " | ".join(parts)

    def generate_head_to_head_document(
        self,
        race_data: List[Dict[str, Any]],
        driver1_code: str,
        driver2_code: str,
        driver1_name: str,
        driver2_name: str
    ) -> str:
        """Generate head-to-head comparison document"""

        h2h_results = {'driver1_wins': 0, 'driver2_wins': 0, 'both_dnf': 0}

        for race in race_data:
            d1_result = None
            d2_result = None

            for result in race.get('race_results', []):
                if result.get('driver_code') == driver1_code:
                    d1_result = result
                elif result.get('driver_code') == driver2_code:
                    d2_result = result

            if d1_result and d2_result:
                d1_pos = d1_result.get('position')
                d2_pos = d2_result.get('position')

                # Both finished
                if d1_pos and d2_pos:
                    if d1_pos < d2_pos:
                        h2h_results['driver1_wins'] += 1
                    elif d2_pos < d1_pos:
                        h2h_results['driver2_wins'] += 1
                # One DNF
                elif d1_pos and not d2_pos:
                    h2h_results['driver1_wins'] += 1
                elif d2_pos and not d1_pos:
                    h2h_results['driver2_wins'] += 1
                # Both DNF
                else:
                    h2h_results['both_dnf'] += 1

        total_h2h = h2h_results['driver1_wins'] + h2h_results['driver2_wins']

        if total_h2h == 0:
            return f"Head-to-head: {driver1_name} vs {driver2_name} | No direct comparisons available"

        d1_pct = h2h_results['driver1_wins'] / total_h2h * 100

        parts = []
        parts.append(f"Head-to-head: {driver1_name} vs {driver2_name}")
        parts.append(f"{driver1_name}: {h2h_results['driver1_wins']} wins ({d1_pct:.0f}%)")
        parts.append(f"{driver2_name}: {h2h_results['driver2_wins']} wins ({100-d1_pct:.0f}%)")
        parts.append(f"Total comparisons: {total_h2h}")

        return " | ".join(parts)


class F1FantasyIngestion:
    """Orchestrate fantasy-optimized data ingestion"""

    def __init__(
        self,
        csv_data_dir: str = "./data/archive",
        vector_db: Optional[F1VectorDB] = None
    ):
        self.loader = F1CSVDataLoader(data_dir=csv_data_dir)
        self.processor = F1FantasyDataProcessor(self.loader)
        self.vector_db = vector_db or F1VectorDB()

        logger.info("Fantasy ingestion initialized")

    def ingest_fantasy_data(
        self,
        start_year: Optional[int] = 2020,
        end_year: Optional[int] = None,
        include_head_to_head: bool = True,
        batch_size: int = 50
    ):
        """
        Ingest fantasy-optimized F1 data

        Args:
            start_year: Starting year (default: 2020 for recent relevant data)
            end_year: Ending year (default: None = latest)
            include_head_to_head: Generate head-to-head comparisons
            batch_size: Batch size for vector uploads
        """
        logger.info("=" * 60)
        logger.info("F1 FANTASY DATA INGESTION")
        logger.info("=" * 60)

        # Load base race data
        logger.info(f"\n[1/4] Loading base race data...")
        race_data = self.loader.load_all_races(
            start_year=start_year,
            end_year=end_year,
            include_pit_stops=True
        )

        logger.info(f"✓ Loaded {len(race_data)} races")

        # Ingest base race data first
        logger.info(f"\n[2/4] Ingesting base race data...")
        self.vector_db.ingest_race_data(race_data, batch_size=batch_size)

        # Generate fantasy-specific documents
        logger.info(f"\n[3/4] Generating fantasy-specific documents...")
        fantasy_vectors = self._generate_fantasy_vectors(race_data, include_head_to_head)

        logger.info(f"✓ Generated {len(fantasy_vectors)} fantasy documents")

        # Ingest fantasy documents
        logger.info(f"\n[4/4] Ingesting fantasy documents...")
        self._ingest_fantasy_vectors(fantasy_vectors, batch_size)

        # Get final stats
        stats = self.vector_db.get_stats()

        logger.info("\n" + "=" * 60)
        logger.info("FANTASY INGESTION COMPLETE!")
        logger.info("=" * 60)

        print(f"\n📊 SUMMARY:")
        print(f"   • Races processed: {len(race_data)}")
        print(f"   • Fantasy documents: {len(fantasy_vectors)}")
        print(f"   • Total vectors: {stats['total_vectors']:,}")
        print(f"   • Year range: {start_year or 'all'} - {end_year or 'latest'}")

    def _generate_fantasy_vectors(
        self,
        race_data: List[Dict[str, Any]],
        include_head_to_head: bool
    ) -> List[Dict[str, Any]]:
        """Generate all fantasy-specific vector documents"""
        vectors = []

        # Get unique drivers and circuits from data
        drivers = {}  # {code: (name, team)}
        circuits = set()

        for race in race_data:
            circuits.add(race['circuit_name'])
            for result in race.get('race_results', []):
                driver_code = result.get('driver_code')
                if driver_code:
                    drivers[driver_code] = (
                        result.get('driver_name'),
                        result.get('team')
                    )

        logger.info(f"Found {len(drivers)} unique drivers, {len(circuits)} circuits")

        # Generate driver performance documents
        logger.info("Generating driver performance documents...")
        for driver_code, (driver_name, team) in drivers.items():
            if not driver_code or not driver_name:
                continue

            # Overall performance document
            doc_text = self.processor.generate_fantasy_driver_document(
                race_data, driver_code, driver_name, team
            )

            if doc_text:
                vectors.append({
                    'id': f"fantasy_driver_{driver_code}_overall",
                    'text': doc_text,
                    'metadata': {
                        'type': 'fantasy_driver_profile',
                        'driver_code': driver_code,
                        'driver_name': driver_name,
                        'team': team,
                        'data_type': 'overall_performance'
                    }
                })

            # Circuit-specific documents
            for circuit in circuits:
                circuit_doc = self.processor.generate_fantasy_driver_document(
                    race_data, driver_code, driver_name, team, upcoming_circuit=circuit
                )

                if circuit_doc:
                    vectors.append({
                        'id': f"fantasy_driver_{driver_code}_{circuit.replace(' ', '_')}",
                        'text': circuit_doc,
                        'metadata': {
                            'type': 'fantasy_driver_circuit',
                            'driver_code': driver_code,
                            'driver_name': driver_name,
                            'team': team,
                            'circuit': circuit,
                            'data_type': 'circuit_performance'
                        }
                    })

        # Generate circuit analysis documents
        logger.info("Generating circuit analysis documents...")
        for circuit in circuits:
            circuit_doc = self.processor.generate_fantasy_circuit_document(race_data, circuit)

            if circuit_doc:
                vectors.append({
                    'id': f"fantasy_circuit_{circuit.replace(' ', '_')}",
                    'text': circuit_doc,
                    'metadata': {
                        'type': 'fantasy_circuit_analysis',
                        'circuit': circuit,
                        'data_type': 'circuit_characteristics'
                    }
                })

        # Generate head-to-head documents
        if include_head_to_head:
            logger.info("Generating head-to-head comparisons...")
            driver_list = list(drivers.items())

            # Generate for teammates and close competitors
            for i, (d1_code, (d1_name, d1_team)) in enumerate(driver_list):
                for d2_code, (d2_name, d2_team) in driver_list[i+1:]:
                    # Only generate for teammates or if both drivers have significant data
                    if d1_team == d2_team or (d1_code and d2_code):
                        h2h_doc = self.processor.generate_head_to_head_document(
                            race_data, d1_code, d2_code, d1_name, d2_name
                        )

                        if "No direct comparisons" not in h2h_doc:
                            vectors.append({
                                'id': f"fantasy_h2h_{d1_code}_vs_{d2_code}",
                                'text': h2h_doc,
                                'metadata': {
                                    'type': 'fantasy_head_to_head',
                                    'driver1': d1_code,
                                    'driver2': d2_code,
                                    'data_type': 'head_to_head_comparison'
                                }
                            })

        return vectors

    def _ingest_fantasy_vectors(self, vectors: List[Dict[str, Any]], batch_size: int):
        """Ingest fantasy vectors into the database"""
        from tqdm import tqdm

        vectors_to_upsert = []

        for vector_data in tqdm(vectors, desc="Processing fantasy docs"):
            # Generate embedding
            embedding = self.vector_db.embedder.generate_embedding(vector_data['text'])

            vectors_to_upsert.append({
                'id': vector_data['id'],
                'values': embedding,
                'metadata': {
                    **vector_data['metadata'],
                    'text': vector_data['text']
                }
            })

            # Batch upsert
            if len(vectors_to_upsert) >= batch_size:
                self.vector_db.index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []

        # Upsert remaining
        if vectors_to_upsert:
            self.vector_db.index.upsert(vectors=vectors_to_upsert)


def main():
    """Main execution"""
    import argparse

    load_dotenv()

    parser = argparse.ArgumentParser(description='Ingest F1 fantasy-optimized data')
    parser.add_argument('--start-year', type=int, default=2020,
                        help='Starting year (default: 2020)')
    parser.add_argument('--end-year', type=int, default=None,
                        help='Ending year (default: latest)')
    parser.add_argument('--no-head-to-head', action='store_true',
                        help='Skip head-to-head comparisons')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size (default: 50)')

    args = parser.parse_args()

    ingestion = F1FantasyIngestion(csv_data_dir='./data/archive')

    ingestion.ingest_fantasy_data(
        start_year=args.start_year,
        end_year=args.end_year,
        include_head_to_head=not args.no_head_to_head,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
