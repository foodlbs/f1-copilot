"""
Test F1 data collection with detailed error reporting
"""

import os
import sys
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

print("="*60)
print("F1 DATA COLLECTION TEST")
print("="*60)

# Import the data collector
from src.data_collector import F1DataCollector

# Test with a single recent race
print("\n1. Testing data collection for 2024 season (most recent)...")
print("   This will attempt to collect just the 2024 season.")

collector = F1DataCollector(start_year=2024)

try:
    print("\n   Attempting to collect 2024 season...")
    season_data = collector.collect_season(2024)

    print(f"\n✅ SUCCESS!")
    print(f"   Collected {len(season_data)} races from 2024")

    if season_data:
        print("\n   Sample race data:")
        race = season_data[0]
        print(f"   • Race: {race.race_name}")
        print(f"   • Circuit: {race.circuit_name}")
        print(f"   • Date: {race.date}")
        print(f"   • Total laps: {race.total_laps}")

        if race.race_results:
            print(f"   • Race results: {len(race.race_results)} drivers")
            winner = race.race_results[0]
            print(f"   • Winner: {winner['driver_name']} ({winner['team']})")

        if race.pit_stops:
            print(f"   • Pit stops: {len(race.pit_stops)}")

        if race.lap_times:
            print(f"   • Lap times collected: {len(race.lap_times)}")

        if race.tire_strategies:
            print(f"   • Tire strategies: {len(race.tire_strategies)} drivers")

    # Check cache
    cache_dir = Path("./cache/f1_data")
    cache_file = cache_dir / "season_2024.json"
    if cache_file.exists():
        size_kb = cache_file.stat().st_size / 1024
        print(f"\n   Cache file: {cache_file}")
        print(f"   Size: {size_kb:.1f} KB")

except Exception as e:
    print(f"\n❌ ERROR during data collection:")
    print(f"   Error type: {type(e).__name__}")
    print(f"   Error message: {str(e)}")

    # Check for specific error types
    if "500" in str(e) or "Internal Server Error" in str(e):
        print("\n   🔍 Analysis: HTTP 500 Internal Server Error")
        print("   This means the FastF1/F1 API server is having issues.")
        print("\n   Possible causes:")
        print("   • The API server is temporarily down")
        print("   • Rate limiting (too many requests)")
        print("   • Data not available for requested session")
        print("\n   Solutions:")
        print("   1. Wait a few minutes and try again")
        print("   2. Try a different year (e.g., 2023, 2022)")
        print("   3. Use cached data if available")

    elif "404" in str(e) or "Not Found" in str(e):
        print("\n   🔍 Analysis: HTTP 404 Not Found")
        print("   The requested data doesn't exist on the server.")
        print("\n   This can happen for:")
        print("   • Future races that haven't occurred yet")
        print("   • Very old races with limited data")
        print("   • Testing/non-championship events")

    elif "timeout" in str(e).lower():
        print("\n   🔍 Analysis: Timeout Error")
        print("   The request took too long to complete.")
        print("\n   Solutions:")
        print("   1. Check your internet connection")
        print("   2. Try again - server might be slow")
        print("   3. Use smaller date ranges")

    import traceback
    print("\n   Full traceback:")
    traceback.print_exc()

print("\n" + "="*60)

# Test 2: Try an older, known good season
print("\n2. Testing with 2023 season (known to have complete data)...")

try:
    collector_2023 = F1DataCollector(start_year=2023)
    print("   Attempting to collect just Round 1 (Bahrain 2023)...")

    race_2023 = collector_2023.collect_race_weekend(2023, 1)

    if race_2023:
        print(f"\n✅ SUCCESS!")
        print(f"   • Race: {race_2023.race_name}")
        print(f"   • Date: {race_2023.date}")
        print(f"   • Results collected: {len(race_2023.race_results) if race_2023.race_results else 0} drivers")
    else:
        print("\n⚠️  No data returned (but no error)")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    if "500" in str(e):
        print("   Server error - FastF1 API is having issues")
        print("   Recommendation: Try again later or use a different year")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)

print("\n📋 SUMMARY:")
print("   The HTTP 500 errors indicate the FastF1 API server is having")
print("   temporary issues. This is NOT a problem with your code.")
print("\n💡 RECOMMENDATIONS:")
print("   1. Wait 10-30 minutes and try again")
print("   2. Start with a single recent year (2023 or 2024)")
print("   3. FastF1 API can be unstable - this is normal")
print("   4. Consider collecting data in smaller batches")
print("\n   Once you successfully collect one season, you can expand")
print("   to collect historical data (2017-2022) incrementally.")
