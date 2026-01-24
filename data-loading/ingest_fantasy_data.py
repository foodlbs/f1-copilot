#!/usr/bin/env python3
"""
F1 Fantasy Data Ingestion - Quick Start

Ingest fantasy-optimized F1 data for lineup recommendation systems.

Includes:
- Driver performance trends and consistency
- Circuit-specific historical performance
- Recent form and momentum analysis
- Head-to-head driver comparisons
- Constructor reliability metrics
- Qualifying importance analysis
- Overtaking potential by circuit

Usage:
    # Recommended: Recent data (2020-2024)
    python ingest_fantasy_data.py

    # Just 2024 season
    python ingest_fantasy_data.py --year 2024

    # Custom range
    python ingest_fantasy_data.py --start-year 2018 --end-year 2023
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
        print("❌ ERROR: OPENAI_API_KEY not set in .env file")
        return False

    if not pinecone_key:
        print("❌ ERROR: PINECONE_API_KEY not set in .env file")
        return False

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Ingest F1 fantasy-optimized data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest_fantasy_data.py                      # Recent data (2020-2024)
  python ingest_fantasy_data.py --year 2024          # Just 2024 season
  python ingest_fantasy_data.py --start-year 2018    # 2018 onwards
        """
    )

    parser.add_argument(
        '--year',
        type=int,
        help='Ingest a single year'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        help='Starting year (default: 2020)'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        help='Ending year (default: latest)'
    )
    parser.add_argument(
        '--no-head-to-head',
        action='store_true',
        help='Skip head-to-head comparisons (faster)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Batch size for uploads (default: 50)'
    )

    args = parser.parse_args()

    # Check environment
    print("=" * 60)
    print("F1 FANTASY DATA INGESTION")
    print("=" * 60)

    if not check_environment():
        sys.exit(1)

    print("✓ Environment configured\n")

    # Determine year range
    if args.year:
        start_year = args.year
        end_year = args.year
        print(f"Mode: Single season ({args.year})")
    else:
        start_year = args.start_year or 2020
        end_year = args.end_year
        print(f"Mode: Recent data ({start_year}-{end_year or 'latest'})")

    print(f"\nFantasy features:")
    print(f"  • Driver performance trends")
    print(f"  • Circuit-specific analysis")
    print(f"  • Recent form & momentum")
    print(f"  • Head-to-head comparisons: {'Yes' if not args.no_head_to_head else 'No'}")

    print("\nStarting ingestion...\n")

    # Import and run
    try:
        from fantasy_data_ingestion import F1FantasyIngestion

        ingestion = F1FantasyIngestion(csv_data_dir='./data/archive')

        ingestion.ingest_fantasy_data(
            start_year=start_year,
            end_year=end_year,
            include_head_to_head=not args.no_head_to_head,
            batch_size=args.batch_size
        )

        print("\n✅ Fantasy data ingestion completed!")
        print("\nYour vector database now includes:")
        print("  ✓ Driver performance profiles")
        print("  ✓ Circuit-specific performance")
        print("  ✓ Recent form analysis")
        print("  ✓ Head-to-head comparisons")
        print("  ✓ Circuit characteristics")
        print("\nReady for fantasy lineup recommendations!")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
