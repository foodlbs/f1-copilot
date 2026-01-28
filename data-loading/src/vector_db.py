"""
F1 Vector Database Module

Manages Pinecone vector database for F1 race knowledge base.
Handles:
- Embedding generation using OpenAI text-embedding-3-large
- Vector storage and retrieval
- Semantic search for race strategies
- Context retrieval for RAG applications
"""

import logging
import os
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time

import numpy as np
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class VectorDBConfig:
    """Configuration for vector database"""
    index_name: str = "f1-race-knowledge"
    dimension: int = 3072  # text-embedding-3-large dimension
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"


class EmbeddingGenerator:
    """Generate embeddings using OpenAI"""

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-large"):
        """
        Initialize embedding generator

        Args:
            api_key: OpenAI API key
            model: Embedding model to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.dimension = 3072 if "large" in model else 1536

        logger.info(f"Embedding generator initialized with {model}")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches

        Args:
            texts: List of input texts
            batch_size: Batch size for API calls

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]

            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )

                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

                # Rate limiting
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in batch {i}: {e}")
                # Add zero vectors for failed batch
                embeddings.extend([[0.0] * self.dimension] * len(batch))

        return embeddings


class F1VectorDB:
    """Manages F1 race data in Pinecone vector database"""

    def __init__(
        self,
        pinecone_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        config: Optional[VectorDBConfig] = None
    ):
        """
        Initialize vector database

        Args:
            pinecone_api_key: Pinecone API key
            openai_api_key: OpenAI API key
            config: Vector DB configuration
        """
        self.config = config or VectorDBConfig()

        # Initialize Pinecone
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key required")

        self.pc = Pinecone(api_key=self.pinecone_api_key)

        # Initialize embedding generator
        self.embedder = EmbeddingGenerator(api_key=openai_api_key)

        # Initialize or get index
        self.index = self._initialize_index()

        logger.info(f"F1 Vector DB initialized with index: {self.config.index_name}")

    def _initialize_index(self):
        """Initialize or retrieve Pinecone index"""
        index_name = self.config.index_name

        # Check if index exists
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if index_name not in existing_indexes:
            logger.info(f"Creating new index: {index_name}")

            self.pc.create_index(
                name=index_name,
                dimension=self.config.dimension,
                metric=self.config.metric,
                spec=ServerlessSpec(
                    cloud=self.config.cloud,
                    region=self.config.region
                )
            )

            # Wait for index to be ready
            time.sleep(5)

        return self.pc.Index(index_name)

    def create_race_document(self, race_data: Dict[str, Any]) -> str:
        """
        Create a searchable text document from race data

        Args:
            race_data: Race weekend data dictionary

        Returns:
            Text representation for embedding
        """
        parts = []

        # Basic info
        parts.append(f"Race: {race_data['race_name']} {race_data['season']}")
        parts.append(f"Circuit: {race_data['circuit_name']}, {race_data['country']}")
        parts.append(f"Date: {race_data['date']}")

        # Race results
        if race_data.get('race_results'):
            winner = race_data['race_results'][0]
            parts.append(f"Winner: {winner['driver_name']} ({winner['team']})")

            podium = [r['driver_name'] for r in race_data['race_results'][:3]]
            parts.append(f"Podium: {', '.join(podium)}")

        # Weather
        if race_data.get('weather_data'):
            weather = race_data['weather_data']
            parts.append(f"Weather: Air {weather['air_temp_avg']:.1f}°C, Track {weather['track_temp_avg']:.1f}°C")
            if weather['rainfall']:
                parts.append("Conditions: WET")

        # Tire strategies
        if race_data.get('tire_strategies'):
            strategies = race_data['tire_strategies']

            # Get winner's strategy
            winner_strat = next((s for s in strategies if s['driver'] == winner.get('driver_code')), None)
            if winner_strat:
                # Handle both API format (with stints) and CSV format (without stints)
                if 'stints' in winner_strat and winner_strat['stints']:
                    compounds = ' → '.join([stint['compound'] for stint in winner_strat['stints']])
                    parts.append(f"Winning strategy: {winner_strat['total_stops']} stop(s), {compounds}")
                else:
                    # CSV format - just show number of stops
                    parts.append(f"Winning strategy: {winner_strat['total_stops']} stop(s)")

        # Pit stops
        if race_data.get('pit_stops'):
            avg_pit_time = np.mean([p['pit_duration'] for p in race_data['pit_stops'] if p.get('pit_duration')])
            parts.append(f"Average pit stop: {avg_pit_time:.2f}s")

        # Incidents
        if race_data.get('safety_cars', 0) > 0:
            parts.append(f"Safety cars: {race_data['safety_cars']}")
        if race_data.get('red_flags', 0) > 0:
            parts.append(f"Red flags: {race_data['red_flags']}")

        # Lap times
        if race_data.get('lap_times'):
            fastest = min(race_data['lap_times'], key=lambda x: x['time'] if x.get('time') else float('inf'))
            if fastest.get('time'):
                parts.append(f"Fastest lap: {fastest['driver']} - {fastest['time']:.3f}s")

        return " | ".join(parts)

    def create_strategy_document(self, race_data: Dict[str, Any], driver_strategy: Dict[str, Any]) -> str:
        """
        Create a detailed strategy document for a specific driver

        Args:
            race_data: Race weekend data
            driver_strategy: Tire strategy for specific driver

        Returns:
            Detailed strategy text for embedding
        """
        parts = []

        parts.append(f"Driver: {driver_strategy['driver']}")
        parts.append(f"Race: {race_data['race_name']} {race_data['season']}")
        parts.append(f"Circuit: {race_data['circuit_name']}")

        # Strategy details
        parts.append(f"Strategy: {driver_strategy['total_stops']} stop(s)")

        # Handle both API format (with stints) and CSV format (with stop_laps)
        if 'stints' in driver_strategy and driver_strategy['stints']:
            for i, stint in enumerate(driver_strategy['stints'], 1):
                parts.append(
                    f"Stint {i}: {stint['compound']} compound, "
                    f"Laps {stint['start_lap']}-{stint['end_lap']} ({stint['stint_length']} laps)"
                )
        elif 'stop_laps' in driver_strategy and driver_strategy['stop_laps']:
            # CSV format - just show stop laps
            stop_laps_str = ', '.join([f"Lap {lap}" for lap in driver_strategy['stop_laps']])
            parts.append(f"Pit stops at: {stop_laps_str}")

        # Get driver's result
        if race_data.get('race_results'):
            result = next((r for r in race_data['race_results'] if r['driver_code'] == driver_strategy['driver']), None)
            if result:
                parts.append(f"Result: P{result['position']} (started P{result['grid_position']})")

        # Weather context
        if race_data.get('weather_data'):
            weather = race_data['weather_data']
            parts.append(f"Conditions: {weather['air_temp_avg']:.1f}°C air, {weather['track_temp_avg']:.1f}°C track")

        return " | ".join(parts)

    def ingest_race_data(self, race_data_list: List[Dict[str, Any]], batch_size: int = 100):
        """
        Ingest F1 race data into vector database

        Args:
            race_data_list: List of race weekend data dictionaries
            batch_size: Batch size for upserts
        """
        logger.info(f"Ingesting {len(race_data_list)} races into vector database...")

        vectors_to_upsert = []

        for race_data in tqdm(race_data_list, desc="Processing races"):
            # Create race overview document
            race_doc = self.create_race_document(race_data)
            race_id = self._generate_id(f"{race_data['season']}-{race_data['round']}-overview")

            race_embedding = self.embedder.generate_embedding(race_doc)

            vectors_to_upsert.append({
                'id': race_id,
                'values': race_embedding,
                'metadata': {
                    'type': 'race_overview',
                    'season': race_data['season'],
                    'round': race_data['round'],
                    'race_name': race_data['race_name'],
                    'circuit': race_data['circuit_name'],
                    'date': race_data['date'],
                    'text': race_doc,
                    'data': json.dumps(race_data)[:40000]  # Pinecone metadata limit
                }
            })

            # Create strategy documents for each driver
            if race_data.get('tire_strategies'):
                for strategy in race_data['tire_strategies']:
                    strategy_doc = self.create_strategy_document(race_data, strategy)
                    strategy_id = self._generate_id(f"{race_data['season']}-{race_data['round']}-{strategy['driver']}")

                    strategy_embedding = self.embedder.generate_embedding(strategy_doc)

                    vectors_to_upsert.append({
                        'id': strategy_id,
                        'values': strategy_embedding,
                        'metadata': {
                            'type': 'driver_strategy',
                            'season': race_data['season'],
                            'round': race_data['round'],
                            'driver': strategy['driver'],
                            'stops': strategy['total_stops'],
                            'text': strategy_doc,
                            'data': json.dumps({
                                'race': race_data['race_name'],
                                'strategy': strategy
                            })[:40000]
                        }
                    })

            # Upsert in batches
            if len(vectors_to_upsert) >= batch_size:
                self._upsert_batch(vectors_to_upsert)
                vectors_to_upsert = []

        # Upsert remaining vectors
        if vectors_to_upsert:
            self._upsert_batch(vectors_to_upsert)

        logger.info(f"✓ Ingestion complete")

    def _upsert_batch(self, vectors: List[Dict]):
        """Upsert a batch of vectors"""
        try:
            self.index.upsert(vectors=vectors)
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            logger.error(f"Error upserting batch: {e}")

    def search_similar_races(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar races based on query

        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Metadata filters

        Returns:
            List of similar race results
        """
        # Generate query embedding
        query_embedding = self.embedder.generate_embedding(query)

        # Search
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )

        return self._format_results(results)

    def search_similar_strategies(
        self,
        query: str,
        circuit: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for similar race strategies

        Args:
            query: Strategy query
            circuit: Filter by circuit
            top_k: Number of results

        Returns:
            List of similar strategies
        """
        filter_dict = {'type': 'driver_strategy'}

        if circuit:
            filter_dict['circuit'] = circuit

        return self.search_similar_races(query, top_k, filter_dict)

    def get_race_context(
        self,
        current_conditions: Dict[str, Any],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get relevant historical context for current race conditions

        Args:
            current_conditions: Current race conditions (circuit, weather, etc.)
            top_k: Number of similar races to retrieve

        Returns:
            List of relevant historical races
        """
        # Build context query
        query_parts = []

        if current_conditions.get('circuit'):
            query_parts.append(f"Circuit: {current_conditions['circuit']}")

        if current_conditions.get('weather'):
            query_parts.append(f"Weather: {current_conditions['weather']}")

        if current_conditions.get('temperature'):
            query_parts.append(f"Temperature: {current_conditions['temperature']}°C")

        if current_conditions.get('conditions'):
            query_parts.append(f"Conditions: {current_conditions['conditions']}")

        query = " | ".join(query_parts)

        # Search for similar conditions
        return self.search_similar_races(query, top_k)

    def _format_results(self, results) -> List[Dict[str, Any]]:
        """Format Pinecone results"""
        formatted = []

        for match in results.matches:
            formatted.append({
                'id': match.id,
                'score': float(match.score),
                'metadata': match.metadata,
                'text': match.metadata.get('text', '')
            })

        return formatted

    def _generate_id(self, key: str) -> str:
        """Generate unique ID for vector"""
        return hashlib.md5(key.encode()).hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = self.index.describe_index_stats()

        return {
            'total_vectors': stats.total_vector_count,
            'dimension': stats.dimension,
            'index_fullness': stats.index_fullness,
        }


if __name__ == "__main__":
    # Example usage
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Initialize vector DB
    vdb = F1VectorDB()

    # Example search
    results = vdb.search_similar_races(
        query="Monaco street circuit wet conditions safety car",
        top_k=5
    )

    print("\nSearch Results:")
    for result in results:
        print(f"\nScore: {result['score']:.3f}")
        print(f"Text: {result['text'][:200]}...")
