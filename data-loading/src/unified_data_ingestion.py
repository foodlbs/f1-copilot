"""
Unified F1 Data Ingestion System

Combines multiple data sources into a comprehensive database:
1. Local CSV files (1950-2024) - Historical archive
2. FastF1 API (2018-present) - Live data with telemetry
3. Ergast API (2017-present) - Race results and standings
4. OpenF1 API (2023-present) - Real-time data, overtakes, radio

Creates a unified vector database with the most complete F1 data available.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import asdict

from dotenv import load_dotenv
from tqdm import tqdm

try:
    from .csv_data_ingestion import F1CSVDataLoader
    from .data_collector import F1DataCollector, RaceWeekendData
    from .vector_db import F1VectorDB
    from .fantasy_data_ingestion import F1FantasyDataProcessor
    from .openf1_collector import OpenF1Collector
except ImportError:
    from csv_data_ingestion import F1CSVDataLoader
    from data_collector import F1DataCollector, RaceWeekendData
    from vector_db import F1VectorDB
    from fantasy_data_ingestion import F1FantasyDataProcessor
    from openf1_collector import OpenF1Collector

# Configure logging
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'unified_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class UnifiedF1DataIngestion:
    """
    Unified ingestion system combining local CSV and online API data

    Data Sources:
    - CSV files: 1950-2024 (historical, no telemetry)
    - FastF1 API: 2018-2024 (detailed telemetry, live data)
    - OpenF1 API: 2023-2024 (overtakes, real-time, positions)
    - Ergast API: Fallback for missing data

    Strategy:
    - Use CSV for historical data (1950-2017)
    - Use FastF1 for modern data (2018-2022)
    - Use FastF1 + OpenF1 for latest data (2023+)
    - Merge and deduplicate
    - Enrich with fantasy metrics
    """

    def __init__(
        self,
        csv_data_dir: str = "./data/archive",
        cache_dir: str = "./cache/f1_data",
        vector_db: Optional[F1VectorDB] = None
    ):
        """
        Initialize unified ingestion system

        Args:
            csv_data_dir: Directory containing CSV files
            cache_dir: Cache directory for API data
            vector_db: Pre-initialized vector DB (or will create new)
        """
        self.csv_loader = F1CSVDataLoader(data_dir=csv_data_dir)
        self.api_collector = F1DataCollector(cache_dir=cache_dir, start_year=2018)
        self.openf1_collector = OpenF1Collector()
        self.vector_db = vector_db or F1VectorDB()
        self.fantasy_processor = F1FantasyDataProcessor(self.csv_loader)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Unified ingestion system initialized (CSV + FastF1 + OpenF1 + Ergast)")

    def ingest_complete_database(
        self,
        csv_start_year: int = 1950,
        csv_end_year: int = 2017,
        api_start_year: int = 2018,
        api_end_year: Optional[int] = None,
        include_fantasy: bool = True,
        force_redownload: bool = False,
        batch_size: int = 50
    ):
        """
        Ingest complete F1 database from all sources

        Args:
            csv_start_year: Start year for CSV data (default: 1950)
            csv_end_year: End year for CSV data (default: 2017)
            api_start_year: Start year for API data (default: 2018)
            api_end_year: End year for API data (default: None = latest)
            include_fantasy: Generate fantasy metrics (default: True)
            force_redownload: Re-download API data even if cached (default: False)
            batch_size: Batch size for vector uploads (default: 50)
        """
        logger.info("=" * 70)
        logger.info("UNIFIED F1 DATA INGESTION - COMPLETE DATABASE")
        logger.info("=" * 70)

        all_race_data = []

        # Phase 1: Historical CSV data (1950-2017)
        if csv_start_year and csv_end_year:
            logger.info(f"\n[1/4] Loading historical CSV data ({csv_start_year}-{csv_end_year})...")
            csv_data = self._load_csv_data(csv_start_year, csv_end_year)
            all_race_data.extend(csv_data)
            logger.info(f"✓ Loaded {len(csv_data)} races from CSV")

        # Phase 2: Modern API data with telemetry (2018+)
        if api_start_year:
            logger.info(f"\n[2/4] Collecting modern API data ({api_start_year}-{api_end_year or 'latest'})...")

            if force_redownload:
                self.api_collector.force_redownload = True

            api_data = self._load_api_data(api_start_year, api_end_year)
            all_race_data.extend(api_data)
            logger.info(f"✓ Loaded {len(api_data)} races from APIs")

        # Phase 3: Deduplicate and merge
        logger.info(f"\n[3/4] Merging and deduplicating data...")
        unified_data = self._merge_and_deduplicate(all_race_data)
        logger.info(f"✓ Unified dataset: {len(unified_data)} unique races")

        # Save unified dataset
        self._save_unified_data(unified_data)

        # Phase 4: Ingest to vector database
        logger.info(f"\n[4/4] Ingesting to vector database...")

        # Base race data
        logger.info("  → Ingesting base race data...")
        self.vector_db.ingest_race_data(unified_data, batch_size=batch_size)

        # Fantasy-optimized data
        if include_fantasy:
            logger.info("  → Generating fantasy metrics...")
            fantasy_vectors = self._generate_fantasy_data(unified_data)
            logger.info(f"  → Ingesting {len(fantasy_vectors)} fantasy documents...")
            self._ingest_fantasy_vectors(fantasy_vectors, batch_size)

        # Final stats
        stats = self.vector_db.get_stats()

        logger.info("\n" + "=" * 70)
        logger.info("UNIFIED INGESTION COMPLETE!")
        logger.info("=" * 70)

        self._print_summary(unified_data, stats)

    def _load_csv_data(self, start_year: int, end_year: int) -> List[Dict[str, Any]]:
        """Load data from CSV files"""
        self.csv_loader.load_optional_data()

        return self.csv_loader.load_all_races(
            start_year=start_year,
            end_year=end_year,
            include_pit_stops=True,
            include_lap_times=False  # Skip lap times for speed
        )

    def _load_api_data(
        self,
        start_year: int,
        end_year: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Load data from online APIs"""

        # Determine years to collect
        if end_year:
            years = list(range(start_year, end_year + 1))
        else:
            current_year = datetime.now().year
            years = list(range(start_year, current_year + 1))

        # Collect from APIs
        race_data = []

        for year in tqdm(years, desc="Collecting API data"):
            try:
                season_data = self.api_collector.collect_season(year)
                race_data.extend(season_data)
                logger.info(f"  ✓ Year {year}: {len(season_data)} races")
            except Exception as e:
                logger.error(f"  ✗ Year {year} failed: {e}")
                continue

        # Convert to dict format
        race_dicts = [asdict(race) for race in race_data]

        # Enrich with OpenF1 for recent years (2023+)
        race_dicts = self._enrich_with_openf1(race_dicts)

        return race_dicts

    def _merge_and_deduplicate(
        self,
        race_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge and deduplicate races from multiple sources

        Priority: API data > CSV data (API has more detail)
        """
        # Index by season + round
        race_index = {}

        for race in race_data:
            key = (race['season'], race['round'])

            # If we already have this race, keep the one with more data
            if key in race_index:
                existing = race_index[key]

                # API data has telemetry and weather - prefer it
                if race.get('weather_data') and not existing.get('weather_data'):
                    race_index[key] = race
                    logger.debug(f"  → Updated {race['race_name']} {race['season']} with API data")
                elif not race.get('weather_data') and existing.get('weather_data'):
                    # Keep existing (already has API data)
                    pass
                else:
                    # Merge: keep API data but add any missing CSV data
                    merged = self._merge_race_data(existing, race)
                    race_index[key] = merged
            else:
                race_index[key] = race

        # Sort by season and round
        unified = sorted(race_index.values(), key=lambda x: (x['season'], x['round']))

        return unified

    def _enrich_with_openf1(
        self,
        race_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enrich race data with OpenF1 API data (2023+)

        Adds:
        - Overtake counts per driver
        - Position change tracking
        - Detailed pit stop timing
        - Real-time intervals

        Does NOT override existing data, only adds new fields.
        """
        logger.info("  → Enriching with OpenF1 data (2023+)...")

        enriched_data = []

        for race in tqdm(race_data, desc="  OpenF1 enrichment"):
            # Only enrich recent races (2023+)
            if race.get('season', 0) < 2023:
                enriched_data.append(race)
                continue

            try:
                # Get OpenF1 meeting for this race
                meetings = self.openf1_collector.get_meetings(
                    year=race['season'],
                    country_name=race.get('country')
                )

                if not meetings:
                    enriched_data.append(race)
                    continue

                # Find matching meeting by race round
                meeting = None
                for m in meetings:
                    if m.get('meeting_official_name') and race['race_name'] in m.get('meeting_official_name', ''):
                        meeting = m
                        break

                # Fallback: use round number
                if not meeting and race['round'] <= len(meetings):
                    meeting = meetings[race['round'] - 1]

                if not meeting:
                    enriched_data.append(race)
                    continue

                meeting_key = meeting['meeting_key']

                # Get race session
                sessions = self.openf1_collector.get_sessions(
                    meeting_key=meeting_key,
                    session_name='Race'
                )

                if not sessions:
                    enriched_data.append(race)
                    continue

                session_key = sessions[0]['session_key']

                # Enrich with OpenF1 data (adds to existing race dict)
                race_enriched = race.copy()

                # Add overtakes
                overtakes = self.openf1_collector.get_overtakes(session_key)
                if overtakes:
                    # Count overtakes per driver
                    from collections import Counter
                    overtake_counts = Counter([o['overtaking_driver_number'] for o in overtakes])

                    race_enriched['openf1_overtakes'] = overtakes
                    race_enriched['openf1_overtake_counts'] = dict(overtake_counts)
                    race_enriched['openf1_total_overtakes'] = len(overtakes)

                # Add position tracking
                positions = self.openf1_collector.get_positions(session_key)
                if positions:
                    race_enriched['openf1_positions'] = positions[:1000]  # Limit to avoid huge data

                # Add intervals
                intervals = self.openf1_collector.get_intervals(session_key)
                if intervals:
                    race_enriched['openf1_intervals'] = intervals[:500]  # Sample

                # Enhanced pit stop data
                pit_stops_of1 = self.openf1_collector.get_pit_stops(session_key)
                if pit_stops_of1:
                    race_enriched['openf1_pit_stops'] = pit_stops_of1

                enriched_data.append(race_enriched)

                logger.debug(f"  ✓ Enriched {race['race_name']} {race['season']} with OpenF1")

            except Exception as e:
                logger.warning(f"  ✗ OpenF1 enrichment failed for {race.get('race_name', 'Unknown')}: {e}")
                enriched_data.append(race)
                continue

        openf1_count = len([r for r in enriched_data if r.get('openf1_overtakes')])
        logger.info(f"  ✓ Enriched {openf1_count} races with OpenF1 data")

        return enriched_data

    def _merge_race_data(
        self,
        primary: Dict[str, Any],
        secondary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge two race records, preferring primary but filling gaps from secondary"""
        merged = primary.copy()

        # Add missing fields from secondary
        for key, value in secondary.items():
            if key not in merged or not merged[key]:
                if value:  # Only add if secondary has data
                    merged[key] = value
            elif isinstance(value, list) and isinstance(merged[key], list):
                # Merge lists (e.g., pit stops, lap times)
                if len(value) > len(merged[key]):
                    merged[key] = value

        return merged

    def _save_unified_data(self, unified_data: List[Dict[str, Any]]):
        """Save unified dataset to JSON"""
        output_dir = Path("./data")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "f1_unified_complete_database.json"

        with open(output_file, 'w') as f:
            json.dump(unified_data, f, indent=2, default=str)

        logger.info(f"✓ Saved unified database to {output_file}")

    def _generate_fantasy_data(
        self,
        race_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate fantasy-optimized documents"""
        vectors = []

        # Get unique drivers and circuits
        drivers = {}
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

        logger.info(f"  → Found {len(drivers)} drivers, {len(circuits)} circuits")

        # Generate driver profiles
        for driver_code, (driver_name, team) in tqdm(list(drivers.items()), desc="  Generating driver profiles"):
            if not driver_code or not driver_name:
                continue

            # Overall profile
            doc_text = self.fantasy_processor.generate_fantasy_driver_document(
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
                        'team': team
                    }
                })

            # Circuit-specific (only for major circuits to save vectors)
            major_circuits = [
                'Monaco Circuit', 'Silverstone Circuit', 'Monza Circuit',
                'Spa-Francorchamps', 'Suzuka Circuit', 'Interlagos'
            ]

            for circuit in major_circuits:
                if circuit in circuits:
                    circuit_doc = self.fantasy_processor.generate_fantasy_driver_document(
                        race_data, driver_code, driver_name, team, upcoming_circuit=circuit
                    )

                    if circuit_doc and "No historical data" not in circuit_doc:
                        vectors.append({
                            'id': f"fantasy_driver_{driver_code}_{circuit.replace(' ', '_')}",
                            'text': circuit_doc,
                            'metadata': {
                                'type': 'fantasy_driver_circuit',
                                'driver_code': driver_code,
                                'driver_name': driver_name,
                                'circuit': circuit
                            }
                        })

        # Generate circuit analyses
        for circuit in tqdm(circuits, desc="  Generating circuit analyses"):
            circuit_doc = self.fantasy_processor.generate_fantasy_circuit_document(race_data, circuit)

            if circuit_doc:
                vectors.append({
                    'id': f"fantasy_circuit_{circuit.replace(' ', '_')}",
                    'text': circuit_doc,
                    'metadata': {
                        'type': 'fantasy_circuit_analysis',
                        'circuit': circuit
                    }
                })

        return vectors

    def _ingest_fantasy_vectors(self, vectors: List[Dict[str, Any]], batch_size: int):
        """Ingest fantasy vectors"""
        vectors_to_upsert = []

        for vector_data in tqdm(vectors, desc="  Uploading fantasy vectors"):
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

    def _print_summary(self, unified_data: List[Dict[str, Any]], stats: Dict[str, Any]):
        """Print ingestion summary"""
        # Calculate data sources
        csv_races = len([r for r in unified_data if not r.get('weather_data')])
        api_races = len([r for r in unified_data if r.get('weather_data')])

        # Year range
        years = sorted(set(r['season'] for r in unified_data))

        print(f"\n📊 UNIFIED DATABASE SUMMARY:")
        print(f"   • Total races: {len(unified_data)}")
        print(f"   • Year range: {years[0]} - {years[-1]}")
        print(f"   • CSV data: {csv_races} races")
        print(f"   • API data: {api_races} races (with telemetry)")
        print(f"   • Vector database: {stats['total_vectors']:,} vectors")
        print(f"   • Data sources: CSV files + FastF1 API + Ergast API")
        print(f"\n💾 Data saved to: ./data/f1_unified_complete_database.json")

    def update_with_latest(self):
        """Update database with latest races from current season"""
        logger.info("Updating with latest races...")

        current_year = datetime.now().year

        # Get latest races from API
        try:
            current_season = self.api_collector.collect_season(current_year)
            logger.info(f"✓ Found {len(current_season)} races in {current_year}")

            # Convert and ingest
            race_dicts = [asdict(race) for race in current_season]

            # Ingest base data
            self.vector_db.ingest_race_data(race_dicts, batch_size=50)

            # Ingest fantasy data
            fantasy_vectors = self._generate_fantasy_data(race_dicts)
            self._ingest_fantasy_vectors(fantasy_vectors, batch_size=50)

            stats = self.vector_db.get_stats()
            logger.info(f"✓ Update complete. Total vectors: {stats['total_vectors']:,}")

        except Exception as e:
            logger.error(f"Update failed: {e}")


def main():
    """Main execution"""
    import argparse

    load_dotenv()

    parser = argparse.ArgumentParser(
        description='Unified F1 data ingestion from CSV + APIs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete database (1950-latest)
  python -m src.unified_data_ingestion --complete

  # Historical + recent (recommended)
  python -m src.unified_data_ingestion --csv-years 1950-2017 --api-years 2018-2024

  # Only modern data with telemetry
  python -m src.unified_data_ingestion --api-only --api-years 2020-2024

  # Update with latest races
  python -m src.unified_data_ingestion --update-only
        """
    )

    parser.add_argument('--complete', action='store_true',
                        help='Ingest complete database (1950-latest)')
    parser.add_argument('--csv-years', type=str,
                        help='CSV year range (e.g., "1950-2017")')
    parser.add_argument('--api-years', type=str,
                        help='API year range (e.g., "2018-2024")')
    parser.add_argument('--api-only', action='store_true',
                        help='Only use API data (skip CSV)')
    parser.add_argument('--no-fantasy', action='store_true',
                        help='Skip fantasy metrics generation')
    parser.add_argument('--force-redownload', action='store_true',
                        help='Force re-download of API data')
    parser.add_argument('--update-only', action='store_true',
                        help='Only update with latest races')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size for uploads (default: 50)')

    args = parser.parse_args()

    # Initialize
    ingestion = UnifiedF1DataIngestion()

    # Update only
    if args.update_only:
        ingestion.update_with_latest()
        return

    # Parse year ranges
    csv_start, csv_end = None, None
    api_start, api_end = None, None

    if args.complete:
        csv_start, csv_end = 1950, 2017
        api_start, api_end = 2018, None
    elif args.api_only:
        csv_start, csv_end = None, None
        if args.api_years:
            parts = args.api_years.split('-')
            api_start = int(parts[0])
            api_end = int(parts[1]) if len(parts) > 1 else None
        else:
            api_start = 2020
            api_end = None
    else:
        if args.csv_years:
            parts = args.csv_years.split('-')
            csv_start = int(parts[0])
            csv_end = int(parts[1]) if len(parts) > 1 else 2017

        if args.api_years:
            parts = args.api_years.split('-')
            api_start = int(parts[0])
            api_end = int(parts[1]) if len(parts) > 1 else None

    # Run ingestion
    ingestion.ingest_complete_database(
        csv_start_year=csv_start,
        csv_end_year=csv_end,
        api_start_year=api_start,
        api_end_year=api_end,
        include_fantasy=not args.no_fantasy,
        force_redownload=args.force_redownload,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
