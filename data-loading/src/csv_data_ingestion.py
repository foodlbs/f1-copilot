"""
CSV Data Ingestion for F1 Vector Database

This module ingests historical F1 data from CSV files in the archive folder
and processes it into the vector database format.

Data sources (1950-2024):
- races.csv: Race calendar and basic info
- results.csv: Race results and standings
- drivers.csv: Driver information
- constructors.csv: Team information
- circuits.csv: Circuit details
- lap_times.csv: Lap time data
- pit_stops.csv: Pit stop data
- qualifying.csv: Qualifying results
- sprint_results.csv: Sprint race results
- driver_standings.csv: Championship standings
- constructor_standings.csv: Constructor championship
"""

import logging
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from tqdm import tqdm
import json

from dotenv import load_dotenv

try:
    from .vector_db import F1VectorDB
except ImportError:
    from vector_db import F1VectorDB

# Configure logging
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'csv_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class CSVRaceData:
    """Structured race data from CSV files"""
    race_id: int
    season: int
    round: int
    race_name: str
    circuit_name: str
    circuit_location: str
    country: str
    date: str
    circuit_ref: str
    lat: float
    lng: float
    race_results: List[Dict[str, Any]]
    qualifying_results: List[Dict[str, Any]]
    sprint_results: List[Dict[str, Any]]
    lap_times: List[Dict[str, Any]]
    pit_stops: List[Dict[str, Any]]
    driver_standings: List[Dict[str, Any]]
    constructor_standings: List[Dict[str, Any]]


class F1CSVDataLoader:
    """Loads and processes F1 data from CSV files"""

    @staticmethod
    def safe_int(value) -> Optional[int]:
        """Safely convert value to int, handling NaN and \\N"""
        if pd.isna(value):
            return None
        if isinstance(value, str) and value.strip() in ['\\N', '', 'NA', 'NULL']:
            return None
        try:
            return int(float(value))  # Handle decimal strings like "1.0"
        except (ValueError, TypeError):
            return None

    @staticmethod
    def safe_float(value) -> Optional[float]:
        """Safely convert value to float, handling NaN and \\N"""
        if pd.isna(value):
            return None
        if isinstance(value, str) and value.strip() in ['\\N', '', 'NA', 'NULL']:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def safe_str(value) -> Optional[str]:
        """Safely convert value to string, handling NaN and \\N"""
        if pd.isna(value):
            return None
        if isinstance(value, str) and value.strip() in ['\\N', '', 'NA', 'NULL']:
            return None
        return str(value)

    def __init__(self, data_dir: str = "./data/archive"):
        """
        Initialize CSV data loader

        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")

        # Load all CSV files
        # Note: CSV files use \N for NULL values, so we need to handle them
        logger.info(f"Loading CSV files from {data_dir}...")
        na_values = ['\\N', 'NA', 'NULL', '']

        self.races_df = pd.read_csv(self.data_dir / "races.csv", na_values=na_values, keep_default_na=True)
        self.results_df = pd.read_csv(self.data_dir / "results.csv", na_values=na_values, keep_default_na=True)
        self.drivers_df = pd.read_csv(self.data_dir / "drivers.csv", na_values=na_values, keep_default_na=True)
        self.constructors_df = pd.read_csv(self.data_dir / "constructors.csv", na_values=na_values, keep_default_na=True)
        self.circuits_df = pd.read_csv(self.data_dir / "circuits.csv", na_values=na_values, keep_default_na=True)
        self.qualifying_df = pd.read_csv(self.data_dir / "qualifying.csv", na_values=na_values, keep_default_na=True)
        self.status_df = pd.read_csv(self.data_dir / "status.csv", na_values=na_values, keep_default_na=True)

        # Optional data (may be large or sparse)
        self.lap_times_df = None
        self.pit_stops_df = None
        self.sprint_results_df = None
        self.driver_standings_df = None
        self.constructor_standings_df = None

        logger.info(f"✓ Loaded {len(self.races_df)} races from CSV files")

    def load_optional_data(self):
        """Load optional/large datasets"""
        logger.info("Loading optional data files...")
        na_values = ['\\N', 'NA', 'NULL', '']

        try:
            self.lap_times_df = pd.read_csv(self.data_dir / "lap_times.csv", na_values=na_values, keep_default_na=True)
            logger.info(f"✓ Loaded {len(self.lap_times_df)} lap times")
        except Exception as e:
            logger.warning(f"Could not load lap times: {e}")

        try:
            self.pit_stops_df = pd.read_csv(self.data_dir / "pit_stops.csv", na_values=na_values, keep_default_na=True)
            logger.info(f"✓ Loaded {len(self.pit_stops_df)} pit stops")
        except Exception as e:
            logger.warning(f"Could not load pit stops: {e}")

        try:
            self.sprint_results_df = pd.read_csv(self.data_dir / "sprint_results.csv", na_values=na_values, keep_default_na=True)
            logger.info(f"✓ Loaded {len(self.sprint_results_df)} sprint results")
        except Exception as e:
            logger.warning(f"Could not load sprint results: {e}")

        try:
            self.driver_standings_df = pd.read_csv(self.data_dir / "driver_standings.csv", na_values=na_values, keep_default_na=True)
            logger.info(f"✓ Loaded {len(self.driver_standings_df)} driver standings")
        except Exception as e:
            logger.warning(f"Could not load driver standings: {e}")

        try:
            self.constructor_standings_df = pd.read_csv(self.data_dir / "constructor_standings.csv", na_values=na_values, keep_default_na=True)
            logger.info(f"✓ Loaded {len(self.constructor_standings_df)} constructor standings")
        except Exception as e:
            logger.warning(f"Could not load constructor standings: {e}")

    def get_race_results(self, race_id: int) -> List[Dict[str, Any]]:
        """Get race results for a specific race"""
        results = self.results_df[self.results_df['raceId'] == race_id].copy()

        if results.empty:
            return []

        # Join with drivers and constructors
        results = results.merge(
            self.drivers_df[['driverId', 'forename', 'surname', 'code', 'nationality']],
            on='driverId',
            how='left'
        )

        results = results.merge(
            self.constructors_df[['constructorId', 'name', 'nationality']],
            on='constructorId',
            how='left',
            suffixes=('_driver', '_constructor')
        )

        results = results.merge(
            self.status_df[['statusId', 'status']],
            on='statusId',
            how='left'
        )

        # Format results
        formatted_results = []
        for _, row in results.iterrows():
            formatted_results.append({
                'position': self.safe_int(row['position']),
                'grid_position': self.safe_int(row['grid']),
                'driver_name': f"{row['forename']} {row['surname']}",
                'driver_code': self.safe_str(row['code']),
                'team': row['name'],
                'points': self.safe_float(row['points']) or 0.0,
                'laps_completed': self.safe_int(row['laps']) or 0,
                'status': self.safe_str(row['status']) or 'Unknown',
                'time': self.safe_str(row['time']),
                'fastest_lap': self.safe_int(row['fastestLap']),
                'fastest_lap_time': self.safe_str(row['fastestLapTime']),
                'fastest_lap_speed': self.safe_float(row['fastestLapSpeed']),
            })

        return sorted(formatted_results, key=lambda x: x['position'] if x['position'] else 999)

    def get_qualifying_results(self, race_id: int) -> List[Dict[str, Any]]:
        """Get qualifying results for a specific race"""
        quali = self.qualifying_df[self.qualifying_df['raceId'] == race_id].copy()

        if quali.empty:
            return []

        # Join with drivers
        quali = quali.merge(
            self.drivers_df[['driverId', 'forename', 'surname', 'code']],
            on='driverId',
            how='left'
        )

        formatted_quali = []
        for _, row in quali.iterrows():
            formatted_quali.append({
                'position': self.safe_int(row['position']),
                'driver_name': f"{row['forename']} {row['surname']}",
                'driver_code': self.safe_str(row['code']),
                'q1': self.safe_str(row['q1']),
                'q2': self.safe_str(row['q2']),
                'q3': self.safe_str(row['q3']),
            })

        return sorted(formatted_quali, key=lambda x: x['position'] if x['position'] else 999)

    def get_lap_times(self, race_id: int, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get lap times for a specific race (limited to avoid memory issues)"""
        if self.lap_times_df is None:
            return []

        laps = self.lap_times_df[self.lap_times_df['raceId'] == race_id].copy()

        if laps.empty:
            return []

        # Limit to avoid memory issues - get fastest laps for each driver
        laps = laps.sort_values('milliseconds').groupby('driverId').head(5)

        # Join with drivers
        laps = laps.merge(
            self.drivers_df[['driverId', 'code']],
            on='driverId',
            how='left'
        )

        formatted_laps = []
        for _, row in laps.head(limit).iterrows():
            milliseconds = self.safe_float(row['milliseconds'])
            formatted_laps.append({
                'driver': self.safe_str(row['code']),
                'lap': self.safe_int(row['lap']),
                'time': milliseconds / 1000.0 if milliseconds else None,
                'time_str': self.safe_str(row['time']),
            })

        return formatted_laps

    def get_pit_stops(self, race_id: int) -> List[Dict[str, Any]]:
        """Get pit stops for a specific race"""
        if self.pit_stops_df is None:
            return []

        stops = self.pit_stops_df[self.pit_stops_df['raceId'] == race_id].copy()

        if stops.empty:
            return []

        # Join with drivers
        stops = stops.merge(
            self.drivers_df[['driverId', 'code']],
            on='driverId',
            how='left'
        )

        formatted_stops = []
        for _, row in stops.iterrows():
            milliseconds = self.safe_float(row['milliseconds'])
            formatted_stops.append({
                'driver': self.safe_str(row['code']),
                'stop': self.safe_int(row['stop']),
                'lap': self.safe_int(row['lap']),
                'pit_duration': milliseconds / 1000.0 if milliseconds else None,
                'time': self.safe_str(row['time']),
            })

        return formatted_stops

    def get_sprint_results(self, race_id: int) -> List[Dict[str, Any]]:
        """Get sprint race results"""
        if self.sprint_results_df is None:
            return []

        sprint = self.sprint_results_df[self.sprint_results_df['raceId'] == race_id].copy()

        if sprint.empty:
            return []

        # Join with drivers and constructors
        sprint = sprint.merge(
            self.drivers_df[['driverId', 'forename', 'surname', 'code']],
            on='driverId',
            how='left'
        )

        sprint = sprint.merge(
            self.constructors_df[['constructorId', 'name']],
            on='constructorId',
            how='left'
        )

        formatted_sprint = []
        for _, row in sprint.iterrows():
            formatted_sprint.append({
                'position': self.safe_int(row['position']),
                'driver_name': f"{row['forename']} {row['surname']}",
                'driver_code': self.safe_str(row['code']),
                'team': row['name'],
                'points': self.safe_float(row['points']) or 0.0,
            })

        return sorted(formatted_sprint, key=lambda x: x['position'] if x['position'] else 999)

    def get_driver_standings(self, race_id: int) -> List[Dict[str, Any]]:
        """Get driver championship standings after this race"""
        if self.driver_standings_df is None:
            return []

        standings = self.driver_standings_df[self.driver_standings_df['raceId'] == race_id].copy()

        if standings.empty:
            return []

        standings = standings.merge(
            self.drivers_df[['driverId', 'forename', 'surname', 'code']],
            on='driverId',
            how='left'
        )

        formatted_standings = []
        for _, row in standings.iterrows():
            formatted_standings.append({
                'position': self.safe_int(row['position']),
                'driver_name': f"{row['forename']} {row['surname']}",
                'driver_code': self.safe_str(row['code']),
                'points': self.safe_float(row['points']) or 0.0,
                'wins': self.safe_int(row['wins']) or 0,
            })

        return sorted(formatted_standings, key=lambda x: x['position'] if x['position'] else 999)

    def get_constructor_standings(self, race_id: int) -> List[Dict[str, Any]]:
        """Get constructor championship standings after this race"""
        if self.constructor_standings_df is None:
            return []

        standings = self.constructor_standings_df[self.constructor_standings_df['raceId'] == race_id].copy()

        if standings.empty:
            return []

        standings = standings.merge(
            self.constructors_df[['constructorId', 'name']],
            on='constructorId',
            how='left'
        )

        formatted_standings = []
        for _, row in standings.iterrows():
            formatted_standings.append({
                'position': self.safe_int(row['position']),
                'constructor': row['name'],
                'points': self.safe_float(row['points']) or 0.0,
                'wins': self.safe_int(row['wins']) or 0,
            })

        return sorted(formatted_standings, key=lambda x: x['position'] if x['position'] else 999)

    def load_all_races(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        include_lap_times: bool = False,
        include_pit_stops: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Load all race data and convert to vector DB format

        Args:
            start_year: Starting year (inclusive)
            end_year: Ending year (inclusive)
            include_lap_times: Include detailed lap times (increases data size significantly)
            include_pit_stops: Include pit stop data

        Returns:
            List of race data dictionaries ready for vector DB ingestion
        """
        # Filter by year
        races = self.races_df.copy()

        if start_year:
            races = races[races['year'] >= start_year]

        if end_year:
            races = races[races['year'] <= end_year]

        logger.info(f"Processing {len(races)} races (years: {start_year or 'all'}-{end_year or 'all'})...")

        # Load optional data if needed
        if include_lap_times or include_pit_stops:
            self.load_optional_data()

        all_race_data = []

        for _, race_row in tqdm(races.iterrows(), total=len(races), desc="Loading races"):
            race_id = int(race_row['raceId'])

            # Get circuit info
            circuit_info = self.circuits_df[
                self.circuits_df['circuitId'] == race_row['circuitId']
            ].iloc[0] if not self.circuits_df[
                self.circuits_df['circuitId'] == race_row['circuitId']
            ].empty else None

            # Build race data
            race_data = {
                'race_id': race_id,
                'season': self.safe_int(race_row['year']),
                'round': self.safe_int(race_row['round']),
                'race_name': str(race_row['name']),
                'date': str(race_row['date']),
                'circuit_name': str(circuit_info['name']) if circuit_info is not None else 'Unknown',
                'circuit_ref': str(circuit_info['circuitRef']) if circuit_info is not None else 'unknown',
                'country': str(circuit_info['country']) if circuit_info is not None else 'Unknown',
                'location': str(circuit_info['location']) if circuit_info is not None else 'Unknown',
                'lat': self.safe_float(circuit_info['lat']) if circuit_info is not None else None,
                'lng': self.safe_float(circuit_info['lng']) if circuit_info is not None else None,
                'race_results': self.get_race_results(race_id),
                'qualifying_results': self.get_qualifying_results(race_id),
                'sprint_results': self.get_sprint_results(race_id),
                'driver_standings': self.get_driver_standings(race_id),
                'constructor_standings': self.get_constructor_standings(race_id),
            }

            # Add optional data
            if include_lap_times:
                race_data['lap_times'] = self.get_lap_times(race_id, limit=500)

            if include_pit_stops:
                race_data['pit_stops'] = self.get_pit_stops(race_id)

            # Calculate tire strategies from pit stops
            race_data['tire_strategies'] = self._extract_tire_strategies(race_data)

            all_race_data.append(race_data)

        logger.info(f"✓ Loaded {len(all_race_data)} races successfully")
        return all_race_data

    def _extract_tire_strategies(self, race_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract tire strategies from pit stop data

        Note: CSV data doesn't include tire compound info, so we estimate based on stops
        """
        if not race_data.get('pit_stops'):
            return []

        strategies = {}

        for stop in race_data['pit_stops']:
            driver = stop['driver']

            if driver not in strategies:
                strategies[driver] = {
                    'driver': driver,
                    'total_stops': 0,
                    'stop_laps': []
                }

            strategies[driver]['total_stops'] = stop['stop']
            strategies[driver]['stop_laps'].append(stop['lap'])

        return list(strategies.values())


class F1CSVIngestion:
    """Orchestrates CSV data ingestion into vector database"""

    def __init__(
        self,
        csv_data_dir: str = "./data/archive",
        vector_db: Optional[F1VectorDB] = None
    ):
        """
        Initialize CSV ingestion

        Args:
            csv_data_dir: Directory containing CSV files
            vector_db: Pre-initialized vector DB (or will create new one)
        """
        self.loader = F1CSVDataLoader(data_dir=csv_data_dir)
        self.vector_db = vector_db or F1VectorDB()

        logger.info("CSV Ingestion initialized")

    def ingest_all(
        self,
        start_year: Optional[int] = 2010,
        end_year: Optional[int] = None,
        include_lap_times: bool = False,
        include_pit_stops: bool = True,
        batch_size: int = 50
    ):
        """
        Ingest all CSV data into vector database

        Args:
            start_year: Starting year for data (default: 2010 for modern F1)
            end_year: Ending year (default: None = all available)
            include_lap_times: Include lap time data (memory intensive)
            include_pit_stops: Include pit stop data
            batch_size: Batch size for vector DB ingestion
        """
        logger.info("=" * 60)
        logger.info("F1 CSV DATA INGESTION")
        logger.info("=" * 60)

        # Load data from CSV
        logger.info(f"\n[1/2] Loading data from CSV files...")
        race_data = self.loader.load_all_races(
            start_year=start_year,
            end_year=end_year,
            include_lap_times=include_lap_times,
            include_pit_stops=include_pit_stops
        )

        # Save processed data
        output_dir = Path("./data")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"f1_csv_processed_{start_year or 'all'}_{end_year or 'latest'}.json"
        with open(output_file, 'w') as f:
            json.dump(race_data, f, indent=2, default=str)

        logger.info(f"✓ Saved processed data to {output_file}")

        # Ingest into vector database
        logger.info(f"\n[2/2] Ingesting into vector database...")
        self.vector_db.ingest_race_data(race_data, batch_size=batch_size)

        # Get stats
        stats = self.vector_db.get_stats()

        logger.info("\n" + "=" * 60)
        logger.info("INGESTION COMPLETE!")
        logger.info("=" * 60)

        print(f"\n📊 SUMMARY:")
        print(f"   • Races processed: {len(race_data)}")
        print(f"   • Year range: {start_year or 'all'} - {end_year or 'latest'}")
        print(f"   • Vector database: {stats['total_vectors']} vectors indexed")
        print(f"   • Data saved to: {output_file}")


def main():
    """Main execution"""
    import argparse

    # Load environment variables
    load_dotenv()

    # Parse arguments
    parser = argparse.ArgumentParser(description='Ingest F1 CSV data into vector database')
    parser.add_argument('--start-year', type=int, default=2010,
                        help='Starting year (default: 2010)')
    parser.add_argument('--end-year', type=int, default=None,
                        help='Ending year (default: latest available)')
    parser.add_argument('--include-lap-times', action='store_true',
                        help='Include detailed lap time data (memory intensive)')
    parser.add_argument('--include-pit-stops', action='store_true', default=True,
                        help='Include pit stop data (default: True)')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size for vector DB ingestion (default: 50)')
    parser.add_argument('--csv-dir', type=str, default='./data/archive',
                        help='Directory containing CSV files (default: ./data/archive)')

    args = parser.parse_args()

    # Check API keys
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set. Cannot generate embeddings.")
        return

    if not os.getenv("PINECONE_API_KEY"):
        logger.error("PINECONE_API_KEY not set. Cannot access vector database.")
        return

    # Run ingestion
    ingestion = F1CSVIngestion(csv_data_dir=args.csv_dir)

    ingestion.ingest_all(
        start_year=args.start_year,
        end_year=args.end_year,
        include_lap_times=args.include_lap_times,
        include_pit_stops=args.include_pit_stops,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
