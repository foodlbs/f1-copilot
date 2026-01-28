"""
F1 Knowledge Base Builder

Orchestrates the complete pipeline:
1. Collect F1 race data (2017-present)
2. Process and structure data
3. Generate embeddings
4. Ingest into vector database

Run this to build the complete knowledge base.
"""

import logging
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from dotenv import load_dotenv

try:
    from .data_collector import F1DataCollector, RaceWeekendData
    from .vector_db import F1VectorDB, VectorDBConfig
except ImportError:
    from data_collector import F1DataCollector, RaceWeekendData
    from vector_db import F1VectorDB, VectorDBConfig

# Configure logging
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'knowledge_base_build.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class F1KnowledgeBaseBuilder:
    """Builds complete F1 knowledge base from scratch"""

    def __init__(
        self,
        start_year: int = 2017,
        data_dir: str = "./data",
        cache_dir: str = "./cache/f1_data",
        force_redownload: bool = False
    ):
        """
        Initialize knowledge base builder

        Args:
            start_year: Starting year for data collection
            data_dir: Directory for processed data
            cache_dir: Directory for caching raw data
            force_redownload: If True, ignore cached data and re-download everything
        """
        self.start_year = start_year
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.force_redownload = force_redownload

        # Initialize components
        self.collector = F1DataCollector(
            cache_dir=str(self.cache_dir),
            start_year=start_year,
            force_redownload=force_redownload
        )

        self.vector_db = None  # Initialized after data collection

        if force_redownload:
            logger.info(f"Knowledge Base Builder initialized (Years: {start_year}-present) - FORCE REDOWNLOAD")
        else:
            logger.info(f"Knowledge Base Builder initialized (Years: {start_year}-present)")

    def build_complete_knowledge_base(
        self,
        years: Optional[List[int]] = None,
        skip_collection: bool = False,
        skip_ingestion: bool = False
    ):
        """
        Build complete knowledge base from scratch

        Args:
            years: Specific years to process (None = all from start_year)
            skip_collection: Skip data collection (use existing data)
            skip_ingestion: Skip vector DB ingestion (only collect data)
        """
        logger.info("=" * 60)
        logger.info("F1 KNOWLEDGE BASE BUILDER")
        logger.info("=" * 60)

        # Step 1: Collect race data
        if not skip_collection:
            logger.info("\n[1/3] Collecting F1 race data...")
            race_data = self._collect_data(years)
        else:
            logger.info("\n[1/3] Loading existing race data...")
            race_data = self._load_existing_data()

        if not race_data:
            logger.error("No race data available!")
            return

        # Step 2: Process and export data
        logger.info("\n[2/3] Processing and exporting data...")
        self._export_data(race_data)

        # Step 3: Ingest into vector database
        if not skip_ingestion:
            logger.info("\n[3/3] Ingesting into vector database...")
            self._ingest_to_vector_db(race_data)
        else:
            logger.info("\n[3/3] Skipping vector database ingestion")

        logger.info("\n" + "=" * 60)
        logger.info("KNOWLEDGE BASE BUILD COMPLETE!")
        logger.info("=" * 60)
        self._print_summary(race_data)

    def _collect_data(self, years: Optional[List[int]] = None) -> List[RaceWeekendData]:
        """Collect F1 race data"""
        try:
            race_data = self.collector.collect_all_seasons(years=years)
            logger.info(f"✓ Collected {len(race_data)} race weekends")
            return race_data

        except Exception as e:
            logger.error(f"Error collecting data: {e}")
            raise

    def _load_existing_data(self) -> List[RaceWeekendData]:
        """Load previously collected data"""
        all_data = []

        # Load from cache
        cache_files = sorted(self.cache_dir.glob("season_*.json"))

        for cache_file in cache_files:
            try:
                with open(cache_file, 'r') as f:
                    season_data = json.load(f)
                    all_data.extend([RaceWeekendData(**race) for race in season_data])
            except Exception as e:
                logger.error(f"Error loading {cache_file}: {e}")

        logger.info(f"✓ Loaded {len(all_data)} races from cache")
        return all_data

    def _export_data(self, race_data: List[RaceWeekendData]):
        """Export processed data to various formats"""
        from dataclasses import asdict

        # Export complete dataset
        complete_file = self.data_dir / "f1_complete_dataset.json"
        with open(complete_file, 'w') as f:
            json.dump([asdict(race) for race in race_data], f, indent=2)

        logger.info(f"✓ Exported complete dataset: {complete_file}")

        # Export summary statistics
        stats = self._generate_statistics(race_data)
        stats_file = self.data_dir / "f1_dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"✓ Exported statistics: {stats_file}")

    def _generate_statistics(self, race_data: List[RaceWeekendData]) -> Dict[str, Any]:
        """Generate dataset statistics"""
        total_races = len(race_data)
        years = sorted(set(race.season for race in race_data))

        total_pit_stops = sum(len(race.pit_stops) for race in race_data if race.pit_stops)
        total_lap_times = sum(len(race.lap_times) for race in race_data if race.lap_times)

        circuits = set()
        drivers = set()
        teams = set()

        for race in race_data:
            circuits.add(race.circuit_name)

            if race.race_results:
                for result in race.race_results:
                    drivers.add(result['driver_name'])
                    teams.add(result['team'])

        return {
            'total_races': total_races,
            'years': years,
            'year_range': f"{years[0]}-{years[-1]}",
            'total_circuits': len(circuits),
            'total_drivers': len(drivers),
            'total_teams': len(teams),
            'total_pit_stops': total_pit_stops,
            'total_lap_times': total_lap_times,
            'circuits': sorted(list(circuits)),
            'generated_at': datetime.now().isoformat()
        }

    def _ingest_to_vector_db(self, race_data: List[RaceWeekendData]):
        """Ingest data into vector database"""
        from dataclasses import asdict

        try:
            # Initialize vector DB
            if self.vector_db is None:
                self.vector_db = F1VectorDB()

            # Convert to dictionaries
            race_dicts = [asdict(race) for race in race_data]

            # Ingest
            self.vector_db.ingest_race_data(race_dicts, batch_size=50)

            # Get stats
            stats = self.vector_db.get_stats()
            logger.info(f"✓ Vector DB stats: {stats['total_vectors']} vectors indexed")

        except Exception as e:
            logger.error(f"Error ingesting to vector DB: {e}")
            raise

    def _print_summary(self, race_data: List[RaceWeekendData]):
        """Print build summary"""
        print(f"\n📊 SUMMARY:")
        print(f"   • Total races: {len(race_data)}")
        print(f"   • Years: {race_data[0].season} - {race_data[-1].season}")
        print(f"   • First race: {race_data[0].race_name} ({race_data[0].date})")
        print(f"   • Last race: {race_data[-1].race_name} ({race_data[-1].date})")

        total_strategies = sum(len(r.tire_strategies) for r in race_data if r.tire_strategies)
        print(f"   • Total strategies indexed: {total_strategies}")

        if self.vector_db:
            stats = self.vector_db.get_stats()
            print(f"   • Vector database: {stats['total_vectors']} vectors")

        print(f"\n💾 Data exported to: {self.data_dir}")

    def update_with_latest_races(self):
        """Update knowledge base with latest races (incremental update)"""
        logger.info("Checking for new races to add...")

        # Load existing data
        existing_data = self._load_existing_data()

        if not existing_data:
            logger.warning("No existing data found. Run full build first.")
            return

        # Get latest season/round
        latest_race = max(existing_data, key=lambda r: (r.season, r.round))
        logger.info(f"Latest race in DB: {latest_race.race_name} ({latest_race.season} R{latest_race.round})")

        # Collect only current season
        current_year = datetime.now().year
        current_season = self.collector.collect_season(current_year)

        # Filter new races
        new_races = [
            race for race in current_season
            if race.season > latest_race.season or
            (race.season == latest_race.season and race.round > latest_race.round)
        ]

        if not new_races:
            logger.info("No new races to add")
            return

        logger.info(f"Found {len(new_races)} new races")

        # Export new data
        self._export_data(existing_data + new_races)

        # Ingest to vector DB
        if self.vector_db or os.getenv("PINECONE_API_KEY"):
            self._ingest_to_vector_db(new_races)

        logger.info("✓ Knowledge base updated")


def main():
    """Main execution"""
    import argparse

    # Load environment variables
    load_dotenv()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Build F1 Knowledge Base')
    parser.add_argument('--start-year', type=int, default=2017,
                        help='Starting year for data collection (default: 2017)')
    parser.add_argument('--skip-collection', action='store_true',
                        help='Skip data collection and use cached data')
    parser.add_argument('--skip-ingestion', action='store_true',
                        help='Skip vector database ingestion (only collect data)')
    parser.add_argument('--force-redownload', action='store_true',
                        help='Ignore cached data and re-download everything')
    parser.add_argument('--update-only', action='store_true',
                        help='Only update with latest races (incremental update)')

    args = parser.parse_args()

    # Check required API keys
    if not os.getenv("OPENAI_API_KEY") and not args.skip_ingestion:
        logger.warning("OPENAI_API_KEY not set. Vector DB ingestion will fail.")

    if not os.getenv("PINECONE_API_KEY") and not args.skip_ingestion:
        logger.warning("PINECONE_API_KEY not set. Vector DB ingestion will be skipped.")

    # Build knowledge base
    builder = F1KnowledgeBaseBuilder(
        start_year=args.start_year,
        force_redownload=args.force_redownload
    )

    if args.update_only:
        # Incremental update
        builder.update_with_latest_races()
    else:
        # Build complete knowledge base
        builder.build_complete_knowledge_base(
            skip_collection=args.skip_collection,
            skip_ingestion=args.skip_ingestion
        )


if __name__ == "__main__":
    main()
