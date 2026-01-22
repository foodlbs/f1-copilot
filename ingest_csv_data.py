#!/usr/bin/env python3
"""
Quick Start Script for CSV Data Ingestion

This script makes it easy to ingest F1 CSV data into your vector database.

Usage:
    # Ingest modern era (2010-2024) - RECOMMENDED
    python ingest_csv_data.py

    # Ingest just 2024 season
    python ingest_csv_data.py --year 2024

    # Ingest specific range
    python ingest_csv_data.py --start-year 2020 --end-year 2023

    # Full historical data (1950-2024)
    python ingest_csv_data.py --all-history
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def check_environment():
    """Check if required environment variables are set"""
    openai_key = os.getenv('OPENAI_API_KEY')
    pinecone_key = os.getenv('PINECONE_API_KEY')

    if not openai_key:
        print("❌ ERROR: OPENAI_API_KEY not set in .env file")
        print("   Please add your OpenAI API key to .env")
        return False

    if not pinecone_key:
        print("❌ ERROR: PINECONE_API_KEY not set in .env file")
        print("   Please add your Pinecone API key to .env")
        return False

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Ingest F1 CSV data into vector database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest_csv_data.py                        # Modern era (2010-2024)
  python ingest_csv_data.py --year 2024            # Just 2024 season
  python ingest_csv_data.py --start-year 2020      # 2020 onwards
  python ingest_csv_data.py --all-history          # Full history (1950-2024)
        """
    )

    parser.add_argument(
        '--year',
        type=int,
        help='Ingest a single year (e.g. --year 2024)'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        help='Starting year (default: 2010 for modern era)'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        help='Ending year (default: latest available)'
    )
    parser.add_argument(
        '--all-history',
        action='store_true',
        help='Ingest all historical data from 1950-2024'
    )
    parser.add_argument(
        '--include-lap-times',
        action='store_true',
        help='Include detailed lap times (memory intensive)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Batch size for vector DB uploads (default: 50)'
    )

    args = parser.parse_args()

    # Check environment
    print("=" * 60)
    print("F1 CSV DATA INGESTION")
    print("=" * 60)

    if not check_environment():
        sys.exit(1)

    print("✓ Environment variables configured\n")

    # Determine year range
    if args.year:
        start_year = args.year
        end_year = args.year
        print(f"Mode: Single season ({args.year})")
    elif args.all_history:
        start_year = 1950
        end_year = None
        print(f"Mode: Full historical data (1950-2024)")
        print("⚠️  This will process 1,126 races and may take 30-60 minutes")
    else:
        start_year = args.start_year or 2010
        end_year = args.end_year
        print(f"Mode: Modern era ({start_year}-{end_year or 'latest'})")

    print(f"\nSettings:")
    print(f"  • Start year: {start_year}")
    print(f"  • End year: {end_year or 'latest'}")
    print(f"  • Include lap times: {'Yes' if args.include_lap_times else 'No'}")
    print(f"  • Batch size: {args.batch_size}")

    # Confirm
    if args.all_history:
        response = input("\nThis will ingest all historical data. Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            sys.exit(0)

    print("\nStarting ingestion...\n")

    # Import and run ingestion
    try:
        from csv_data_ingestion import F1CSVIngestion

        ingestion = F1CSVIngestion(csv_data_dir='./data/archive')

        ingestion.ingest_all(
            start_year=start_year,
            end_year=end_year,
            include_lap_times=args.include_lap_times,
            include_pit_stops=True,
            batch_size=args.batch_size
        )

        print("\n✅ Ingestion completed successfully!")
        print("\nNext steps:")
        print("  • Check vector DB stats")
        print("  • Test semantic search queries")
        print("  • Use vectors for RAG/prediction models")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
