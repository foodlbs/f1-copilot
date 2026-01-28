#!/usr/bin/env python3
"""
Complete F1 Database Ingestion

Combines ALL data sources into one comprehensive database:
- Local CSV files (1950-2024) - Historical archive
- FastF1 API (2018-2024) - Live data with telemetry
- Ergast API - Fallback data source

Creates the ultimate F1 database for fantasy apps!

Usage:
    # Recommended: Complete database (1950-latest)
    python ingest_complete_database.py

    # Quick test: Modern data only (2020-latest)
    python ingest_complete_database.py --modern-only

    # Update with latest races
    python ingest_complete_database.py --update-only
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dotenv import load_dotenv

load_dotenv()


def check_environment():
    """Check required environment variables"""
    openai_key = os.getenv('OPENAI_API_KEY')
    pinecone_key = os.getenv('PINECONE_API_KEY')

    if not openai_key:
        print("❌ ERROR: OPENAI_API_KEY not set")
        return False

    if not pinecone_key:
        print("❌ ERROR: PINECONE_API_KEY not set")
        return False

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Ingest complete F1 database from all sources',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data Sources:
  • CSV files (1950-2017): Historical races
  • FastF1 API (2018-2024): Modern races with telemetry
  • Ergast API: Fallback data

Examples:
  python ingest_complete_database.py                    # Complete DB (1950-latest)
  python ingest_complete_database.py --modern-only      # 2020-latest only
  python ingest_complete_database.py --update-only      # Latest races only
        """
    )

    parser.add_argument(
        '--modern-only',
        action='store_true',
        help='Only ingest modern data (2020-latest) for testing'
    )
    parser.add_argument(
        '--no-fantasy',
        action='store_true',
        help='Skip fantasy metrics (faster)'
    )
    parser.add_argument(
        '--update-only',
        action='store_true',
        help='Only update with latest races from current season'
    )
    parser.add_argument(
        '--force-redownload',
        action='store_true',
        help='Re-download API data even if cached'
    )

    args = parser.parse_args()

    # Check environment
    print("=" * 70)
    print("F1 COMPLETE DATABASE INGESTION")
    print("=" * 70)

    if not check_environment():
        sys.exit(1)

    print("✓ Environment configured\n")

    # Import here to avoid loading if env check fails
    from unified_data_ingestion import UnifiedF1DataIngestion

    ingestion = UnifiedF1DataIngestion()

    # Update only
    if args.update_only:
        print("Mode: Update with latest races\n")
        print("This will:")
        print("  • Fetch latest races from current season")
        print("  • Update vector database")
        print("  • Fast (5-10 minutes)\n")

        ingestion.update_with_latest()
        print("\n✅ Update complete!")
        return

    # Determine mode
    if args.modern_only:
        print("Mode: Modern data only (2020-latest)")
        print("\nThis will:")
        print("  • Load CSV data: SKIPPED")
        print("  • Load API data: 2020-latest (with telemetry)")
        print("  • Fantasy metrics: " + ("Included" if not args.no_fantasy else "Skipped"))
        print(f"  • Time: ~20-30 minutes")
        print(f"  • Cost: ~$3-4\n")

        csv_start = None
        csv_end = None
        api_start = 2020
        api_end = None
    else:
        print("Mode: Complete database (1950-latest)")
        print("\nThis will:")
        print("  • Load CSV data: 1950-2017 (historical)")
        print("  • Load API data: 2018-latest (with telemetry)")
        print("  • Fantasy metrics: " + ("Included" if not args.no_fantasy else "Skipped"))
        print(f"  • Time: ~45-60 minutes")
        print(f"  • Cost: ~$10-12\n")

        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            sys.exit(0)

        csv_start = 1950
        csv_end = 2017
        api_start = 2018
        api_end = None

    print("\nStarting ingestion...\n")

    # Run ingestion
    try:
        ingestion.ingest_complete_database(
            csv_start_year=csv_start,
            csv_end_year=csv_end,
            api_start_year=api_start,
            api_end_year=api_end,
            include_fantasy=not args.no_fantasy,
            force_redownload=args.force_redownload,
            batch_size=50
        )

        print("\n✅ Complete database ingestion successful!")
        print("\nYour vector database now includes:")
        print("  ✓ Historical races (CSV)")
        print("  ✓ Modern races with telemetry (API)")
        print("  ✓ Driver performance profiles")
        print("  ✓ Circuit characteristics")
        print("  ✓ Fantasy recommendations")
        print("\nData saved to: ./data/f1_unified_complete_database.json")
        print("\nReady to build your fantasy app! 🏎️🏆")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
