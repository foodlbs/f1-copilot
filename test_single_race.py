"""
Test collecting a single race to verify everything works
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("="*60)
print("TESTING SINGLE RACE COLLECTION")
print("="*60)

from src.data_collector import F1DataCollector

# Initialize collector
collector = F1DataCollector(start_year=2023)

print("\nAttempting to collect 2023 Bahrain GP (Round 1)...")
print("This is a known good race with complete data.\n")

try:
    race = collector.collect_race_weekend(2023, 1)

    if race:
        print(f"\n✅ SUCCESS! Race collected:")
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

        if race.tire_strategies:
            print(f"   • Tire strategies: {len(race.tire_strategies)} drivers")

        if race.lap_times:
            print(f"   • Lap times: {len(race.lap_times)} samples")

        # Check cache was created
        from pathlib import Path
        cache_file = Path("cache/f1_data/race_2023_R1.json")
        if cache_file.exists():
            size_kb = cache_file.stat().st_size / 1024
            print(f"\n   ✓ Race cached: {cache_file}")
            print(f"   ✓ Cache size: {size_kb:.1f} KB")

        print("\n" + "="*60)
        print("TEST PASSED ✅")
        print("="*60)
        print("\nYou can now run the full collection:")
        print("  cd src")
        print("  python knowledge_base_builder.py")

    else:
        print("\n❌ FAILED: No race data returned")
        print("Check the error messages above.")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

    print("\n" + "="*60)
    print("TROUBLESHOOTING")
    print("="*60)
    print("\nPossible causes:")
    print("1. FastF1 API is temporarily down (try again in 5 minutes)")
    print("2. Internet connection issue")
    print("3. The 'FastestLap' bug - this should be fixed now")
    print("\nIf error persists, please share the full error message.")
