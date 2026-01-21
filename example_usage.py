"""
Example Usage of F1 Knowledge Base System

Demonstrates the main features:
1. Building knowledge base
2. Predicting strategies
3. Running race simulations
4. Searching similar races
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def example_1_build_knowledge_base():
    """Example: Build complete knowledge base"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Building Knowledge Base")
    print("="*60)

    from src.knowledge_base_builder import F1KnowledgeBaseBuilder

    builder = F1KnowledgeBaseBuilder(start_year=2022)  # Start with recent years

    # Build knowledge base
    # Set skip_ingestion=True if you just want to collect data without vector DB
    builder.build_complete_knowledge_base(
        skip_collection=False,
        skip_ingestion=False
    )


def example_2_predict_strategy():
    """Example: Predict race strategy"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Predicting Race Strategy")
    print("="*60)

    from src.strategy_predictor import F1StrategyPredictor, SessionData

    # Initialize predictor (will use vector DB)
    predictor = F1StrategyPredictor()

    # Define current race conditions
    session = SessionData(
        circuit="Silverstone",
        session_type="Race",
        lap_number=1,
        total_laps=52,
        air_temp=22.0,
        track_temp=35.0,
        weather="Dry",
        available_compounds=["SOFT", "MEDIUM", "HARD"]
    )

    # Get strategy recommendation
    print(f"\nPredicting strategy for {session.circuit}...")
    recommendation = predictor.predict_optimal_strategy(session)

    print(f"\n✅ RECOMMENDATION:")
    print(f"   Strategy: {recommendation.strategy_type}")
    print(f"   Confidence: {recommendation.confidence:.1%}")
    print(f"\n   Stint Plan:")
    for stint in recommendation.stints:
        pit_info = f"→ Pit on lap {stint['pit_lap']}" if stint['pit_lap'] else "→ Finish"
        print(f"   • Stint {stint['stint_number']}: {stint['compound']} compound")
        print(f"     Laps {stint['start_lap']}-{stint['end_lap']} ({stint['stint_length']} laps) {pit_info}")

    print(f"\n   Reasoning:")
    print(f"   {recommendation.reasoning}")

    # Get pit window prediction
    print(f"\n✅ PIT WINDOW ANALYSIS:")
    earliest, optimal, latest, reasoning = predictor.predict_pit_window(
        session_data=session,
        current_stint_compound="MEDIUM",
        stint_start_lap=1
    )
    print(f"   Earliest: Lap {earliest}")
    print(f"   Optimal: Lap {optimal}")
    print(f"   Latest: Lap {latest}")
    print(f"   {reasoning}")


def example_3_simulate_race():
    """Example: Simulate complete race"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Race Simulation")
    print("="*60)

    from src.race_simulator import RaceSimulator

    # Initialize simulator
    simulator = RaceSimulator(enable_randomness=True)

    # Define driver grid
    drivers = [
        {'number': '1', 'name': 'Max Verstappen', 'team': 'Red Bull', 'base_lap_time': 88.0},
        {'number': '44', 'name': 'Lewis Hamilton', 'team': 'Mercedes', 'base_lap_time': 88.2},
        {'number': '16', 'name': 'Charles Leclerc', 'team': 'Ferrari', 'base_lap_time': 88.3},
        {'number': '11', 'name': 'Sergio Perez', 'team': 'Red Bull', 'base_lap_time': 88.5},
        {'number': '63', 'name': 'George Russell', 'team': 'Mercedes', 'base_lap_time': 88.6},
        {'number': '55', 'name': 'Carlos Sainz', 'team': 'Ferrari', 'base_lap_time': 88.7},
        {'number': '4', 'name': 'Lando Norris', 'team': 'McLaren', 'base_lap_time': 88.9},
        {'number': '81', 'name': 'Oscar Piastri', 'team': 'McLaren', 'base_lap_time': 89.0},
    ]

    # Setup race
    print("\nSetting up race at Spa-Francorchamps...")
    simulator.setup_race(
        circuit="Spa-Francorchamps",
        total_laps=44,
        drivers=drivers,
        weather="Dry",
        air_temp=20.0,
        track_temp=28.0
    )

    print("\nStarting race simulation...")
    print("(This may take a moment...)\n")

    # Run simulation
    results = simulator.simulate_race(verbose=False)

    # Display results
    print("\n" + "="*60)
    print("🏁 FINAL RESULTS")
    print("="*60)

    for classification in results['classifications']:
        gap = f"+{classification['gap_to_leader']:.2f}s" if classification['position'] > 1 else "Winner"
        print(f"P{classification['position']:2d}: {classification['driver']:20s} "
              f"({classification['team']:10s}) - {classification['stops']} stop(s) - {gap}")

    if results['dnf']:
        print(f"\nDNFs:")
        for dnf in results['dnf']:
            print(f"  • {dnf['driver']} - {dnf['reason']}")

    print(f"\n📊 Race Statistics:")
    print(f"   • Total pit stops: {results['statistics']['total_pit_stops']}")
    print(f"   • Average stops per driver: {results['statistics']['average_stops']:.1f}")
    print(f"   • Safety cars: {results['statistics']['safety_cars']}")
    print(f"   • DNFs: {results['statistics']['dnf_count']}")

    # Export lap chart
    lap_chart = simulator.export_lap_chart()
    print(f"\n   • Lap chart exported: {len(lap_chart)} data points")


def example_4_search_similar_races():
    """Example: Search for similar races"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Search Similar Races")
    print("="*60)

    from src.vector_db import F1VectorDB

    # Initialize vector database
    vdb = F1VectorDB()

    # Example searches
    searches = [
        "Street circuit wet conditions safety car",
        "High-speed circuit overtaking opportunities",
        "Monaco tight corners tire management",
    ]

    for query in searches:
        print(f"\n🔍 Searching: '{query}'")
        results = vdb.search_similar_races(query, top_k=3)

        if results:
            print(f"   Found {len(results)} similar races:\n")
            for i, result in enumerate(results, 1):
                metadata = result['metadata']
                print(f"   {i}. {metadata.get('race_name', 'Unknown')} {metadata.get('season', '')}")
                print(f"      Similarity: {result['score']:.3f}")
                print(f"      {result['text'][:150]}...")
                print()
        else:
            print("   No results found")


def example_5_live_race_updates():
    """Example: Live strategy updates during race"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Live Race Strategy Updates")
    print("="*60)

    from src.race_simulator import RaceSimulator
    from src.strategy_predictor import F1StrategyPredictor

    # Initialize
    simulator = RaceSimulator()
    predictor = F1StrategyPredictor()

    # Setup quick race
    drivers = [
        {'number': '1', 'name': 'Max Verstappen', 'team': 'Red Bull', 'base_lap_time': 78.0},
        {'number': '44', 'name': 'Lewis Hamilton', 'team': 'Mercedes', 'base_lap_time': 78.2},
    ]

    simulator.setup_race(
        circuit="Monza",
        total_laps=30,  # Shortened for demo
        drivers=drivers,
        weather="Dry",
        track_temp=35.0
    )

    print("\nSimulating race with live updates...\n")

    # Simulate first 15 laps
    for lap in range(1, 16):
        simulator.simulate_lap()

        if lap % 5 == 0:
            standings = simulator.get_live_standings()
            print(f"LAP {lap}/30:")
            for pos in standings:
                print(f"  P{pos['position']}: {pos['driver']} - "
                      f"{pos['compound']} ({pos['tire_age']} laps)")

    # Update strategy mid-race
    print(f"\n🔄 Updating strategy for Max Verstappen at lap 15...")
    updated = simulator.update_strategy_mid_race("Max Verstappen")
    print(f"   New recommendation: {updated.strategy_type}")

    # Complete race
    while simulator.race_state.current_lap < 30:
        simulator.simulate_lap()

    results = simulator._generate_results()
    print(f"\n🏁 Race Complete!")
    for classification in results['classifications']:
        print(f"   P{classification['position']}: {classification['driver']}")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("F1 KNOWLEDGE BASE - EXAMPLE USAGE")
    print("="*60)

    # Check API keys
    if not os.getenv("PINECONE_API_KEY"):
        print("\n⚠️  Warning: PINECONE_API_KEY not set")
        print("   Some examples will fail. Set API key in .env file.")

    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OPENAI_API_KEY not set")
        print("   Some examples will fail. Set API key in .env file.")

    print("\nSelect example to run:")
    print("1. Build knowledge base (takes time!)")
    print("2. Predict race strategy")
    print("3. Simulate race")
    print("4. Search similar races")
    print("5. Live race updates")
    print("6. Run all examples (except #1)")

    choice = input("\nEnter choice (1-6): ").strip()

    if choice == "1":
        example_1_build_knowledge_base()
    elif choice == "2":
        example_2_predict_strategy()
    elif choice == "3":
        example_3_simulate_race()
    elif choice == "4":
        example_4_search_similar_races()
    elif choice == "5":
        example_5_live_race_updates()
    elif choice == "6":
        example_2_predict_strategy()
        example_3_simulate_race()
        example_4_search_similar_races()
        example_5_live_race_updates()
    else:
        print("Invalid choice")

    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)


if __name__ == "__main__":
    main()
