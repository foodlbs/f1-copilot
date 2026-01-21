"""
Test script to verify setup is working correctly
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("="*60)
print("F1 KNOWLEDGE BASE - SETUP TEST")
print("="*60)

# Test 1: Check environment variables
print("\n1. Testing environment variables...")
pinecone_key = os.getenv("PINECONE_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

if pinecone_key:
    print(f"   ✓ PINECONE_API_KEY found (starts with: {pinecone_key[:10]}...)")
else:
    print("   ✗ PINECONE_API_KEY not found")

if openai_key:
    print(f"   ✓ OPENAI_API_KEY found (starts with: {openai_key[:10]}...)")
else:
    print("   ✗ OPENAI_API_KEY not found")

# Test 2: Import modules
print("\n2. Testing module imports...")
try:
    from src.data_collector import F1DataCollector
    print("   ✓ data_collector module imported")
except Exception as e:
    print(f"   ✗ data_collector import failed: {e}")

try:
    from src.vector_db import F1VectorDB
    print("   ✓ vector_db module imported")
except Exception as e:
    print(f"   ✗ vector_db import failed: {e}")

try:
    from src.strategy_predictor import F1StrategyPredictor
    print("   ✓ strategy_predictor module imported")
except Exception as e:
    print(f"   ✗ strategy_predictor import failed: {e}")

try:
    from src.race_simulator import RaceSimulator
    print("   ✓ race_simulator module imported")
except Exception as e:
    print(f"   ✗ race_simulator import failed: {e}")

# Test 3: Check Pinecone package
print("\n3. Testing Pinecone package...")
try:
    from pinecone import Pinecone, ServerlessSpec
    print("   ✓ Pinecone v8+ imported successfully")

    # Try to connect (but don't create anything)
    if pinecone_key:
        pc = Pinecone(api_key=pinecone_key)
        indexes = list(pc.list_indexes())
        print(f"   ✓ Connected to Pinecone ({len(indexes)} indexes found)")

except Exception as e:
    print(f"   ✗ Pinecone test failed: {e}")

# Test 4: Check OpenAI package
print("\n4. Testing OpenAI package...")
try:
    from openai import OpenAI
    print("   ✓ OpenAI package imported successfully")

    if openai_key:
        client = OpenAI(api_key=openai_key)
        print("   ✓ OpenAI client initialized")

except Exception as e:
    print(f"   ✗ OpenAI test failed: {e}")

# Test 5: Check FastF1
print("\n5. Testing FastF1 package...")
try:
    import fastf1
    print(f"   ✓ FastF1 v{fastf1.__version__} imported successfully")
except Exception as e:
    print(f"   ✗ FastF1 import failed: {e}")

# Test 6: Check directories
print("\n6. Testing directory structure...")
dirs = ['cache/f1_data', 'cache/fastf1', 'data', 'logs', 'src']
for d in dirs:
    if os.path.exists(d):
        print(f"   ✓ {d}/ exists")
    else:
        print(f"   ✗ {d}/ missing")

print("\n" + "="*60)
print("SETUP TEST COMPLETE")
print("="*60)

# Final verdict
if pinecone_key and openai_key:
    print("\n✅ System is ready to use!")
    print("\nNext steps:")
    print("  1. Run: cd src && python knowledge_base_builder.py")
    print("  2. Or try: python example_usage.py")
else:
    print("\n⚠️  Please configure API keys in .env file")
