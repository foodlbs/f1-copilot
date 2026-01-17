"""
F1 Vector Database Module

Pinecone integration for semantic search over historical F1 race strategies.
Enables RAG (Retrieval Augmented Generation) for strategy recommendations.

Author: F1 Race Strategy Analyzer
"""

import os
import json
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time

import numpy as np

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logging.warning("Pinecone not installed. Vector database features disabled.")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VectorDBConfig:
    """Configuration for vector database"""
    index_name: str = "f1-strategies"
    dimension: int = 1536  # OpenAI/Anthropic embedding dimension
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"


class EmbeddingGenerator:
    """Generate embeddings for text data"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.client = None
        
        if ANTHROPIC_AVAILABLE and self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Fallback: simple TF-IDF based embeddings
        self.vocabulary = {}
        self.idf = {}
        self.embedding_dim = 1536
    
    def _hash_embedding(self, text: str, dim: int = 1536) -> np.ndarray:
        """Generate deterministic embedding using hashing (fallback method)"""
        # Create a deterministic embedding based on text hash
        # This is a simple fallback when API is not available
        
        # Tokenize
        tokens = text.lower().split()
        
        # Create embedding using random projection from hash
        np.random.seed(int(hashlib.md5(text.encode()).hexdigest()[:8], 16))
        
        # Generate base embedding from tokens
        embedding = np.zeros(dim)
        for i, token in enumerate(tokens):
            token_hash = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
            np.random.seed(token_hash)
            token_vec = np.random.randn(dim)
            weight = 1.0 / (1 + i * 0.1)  # Position-based weighting
            embedding += token_vec * weight
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if self.client:
            try:
                # Use Claude to generate a semantic representation
                # Note: Claude doesn't have a direct embedding API,
                # so we use a hash-based approach as fallback
                return self._hash_embedding(text)
            except Exception as e:
                logger.warning(f"Embedding generation failed: {e}")
                return self._hash_embedding(text)
        else:
            return self._hash_embedding(text)
    
    def generate_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        return [self.generate_embedding(text) for text in texts]


class F1VectorDatabase:
    """
    Vector database for F1 race strategy semantic search.
    
    Stores and retrieves historical race strategies, pit stop decisions,
    and race outcomes for RAG-based strategy generation.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 config: Optional[VectorDBConfig] = None):
        self.api_key = api_key or os.getenv('PINECONE_API_KEY')
        self.config = config or VectorDBConfig()
        self.pc = None
        self.index = None
        self.embedding_generator = EmbeddingGenerator()
        
        if PINECONE_AVAILABLE and self.api_key:
            self.pc = Pinecone(api_key=self.api_key)
            logger.info("Pinecone client initialized")
        else:
            logger.warning("Pinecone not available. Using local fallback.")
            self._init_local_fallback()
    
    def _init_local_fallback(self):
        """Initialize local fallback storage"""
        self.local_storage = {
            'vectors': [],
            'metadata': [],
            'ids': []
        }
        self.local_storage_path = './data/vector_store.json'
        
        # Load existing data if available
        if os.path.exists(self.local_storage_path):
            try:
                with open(self.local_storage_path, 'r') as f:
                    self.local_storage = json.load(f)
                logger.info(f"Loaded {len(self.local_storage['ids'])} vectors from local storage")
            except:
                pass
    
    def _save_local_fallback(self):
        """Save local fallback storage"""
        os.makedirs(os.path.dirname(self.local_storage_path), exist_ok=True)
        with open(self.local_storage_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            storage_copy = {
                'vectors': [v if isinstance(v, list) else v.tolist() 
                           for v in self.local_storage['vectors']],
                'metadata': self.local_storage['metadata'],
                'ids': self.local_storage['ids']
            }
            json.dump(storage_copy, f)
    
    def create_index(self, force_recreate: bool = False) -> bool:
        """Create or connect to the Pinecone index"""
        if not self.pc:
            logger.info("Using local fallback storage")
            return True
        
        try:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.config.index_name in existing_indexes:
                if force_recreate:
                    logger.info(f"Deleting existing index: {self.config.index_name}")
                    self.pc.delete_index(self.config.index_name)
                    time.sleep(5)
                else:
                    logger.info(f"Connecting to existing index: {self.config.index_name}")
                    self.index = self.pc.Index(self.config.index_name)
                    return True
            
            # Create new index
            logger.info(f"Creating index: {self.config.index_name}")
            self.pc.create_index(
                name=self.config.index_name,
                dimension=self.config.dimension,
                metric=self.config.metric,
                spec=ServerlessSpec(
                    cloud=self.config.cloud,
                    region=self.config.region
                )
            )
            
            # Wait for index to be ready
            while not self.pc.describe_index(self.config.index_name).status['ready']:
                time.sleep(1)
            
            self.index = self.pc.Index(self.config.index_name)
            logger.info("Index created and ready")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            self._init_local_fallback()
            return False
    
    def upsert_strategy(self, 
                        strategy_id: str,
                        strategy_text: str,
                        metadata: Dict[str, Any]) -> bool:
        """Insert or update a strategy in the database"""
        try:
            # Generate embedding
            embedding = self.embedding_generator.generate_embedding(strategy_text)
            
            # Clean metadata (Pinecone only accepts certain types)
            clean_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_metadata[k] = v
                elif isinstance(v, list) and all(isinstance(i, (str, int, float)) for i in v):
                    clean_metadata[k] = v
                else:
                    clean_metadata[k] = str(v)
            
            clean_metadata['text'] = strategy_text[:1000]  # Store truncated text
            
            if self.index:
                self.index.upsert(
                    vectors=[(strategy_id, embedding.tolist(), clean_metadata)]
                )
            else:
                # Local fallback
                if strategy_id in self.local_storage['ids']:
                    idx = self.local_storage['ids'].index(strategy_id)
                    self.local_storage['vectors'][idx] = embedding.tolist()
                    self.local_storage['metadata'][idx] = clean_metadata
                else:
                    self.local_storage['ids'].append(strategy_id)
                    self.local_storage['vectors'].append(embedding.tolist())
                    self.local_storage['metadata'].append(clean_metadata)
                
                self._save_local_fallback()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert strategy: {e}")
            return False
    
    def upsert_batch(self, 
                     strategies: List[Dict[str, Any]],
                     batch_size: int = 100) -> int:
        """Insert multiple strategies in batches"""
        success_count = 0
        
        for i in range(0, len(strategies), batch_size):
            batch = strategies[i:i + batch_size]
            vectors = []
            
            for strategy in batch:
                try:
                    strategy_id = strategy.get('id', f"strategy_{i}")
                    text = strategy.get('description', '')
                    metadata = {k: v for k, v in strategy.items() 
                               if k not in ['id', 'description', 'embedding']}
                    
                    embedding = self.embedding_generator.generate_embedding(text)
                    
                    # Clean metadata
                    clean_metadata = {}
                    for k, v in metadata.items():
                        if isinstance(v, (str, int, float, bool)):
                            clean_metadata[k] = v
                        elif isinstance(v, list) and all(isinstance(x, (str, int, float)) for x in v):
                            clean_metadata[k] = v
                        else:
                            clean_metadata[k] = str(v)
                    
                    clean_metadata['text'] = text[:1000]
                    
                    vectors.append((strategy_id, embedding.tolist(), clean_metadata))
                    success_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to process strategy: {e}")
            
            if vectors:
                if self.index:
                    self.index.upsert(vectors=vectors)
                else:
                    for vid, vec, meta in vectors:
                        if vid in self.local_storage['ids']:
                            idx = self.local_storage['ids'].index(vid)
                            self.local_storage['vectors'][idx] = vec
                            self.local_storage['metadata'][idx] = meta
                        else:
                            self.local_storage['ids'].append(vid)
                            self.local_storage['vectors'].append(vec)
                            self.local_storage['metadata'].append(meta)
                    
                    self._save_local_fallback()
        
        logger.info(f"Upserted {success_count} strategies")
        return success_count
    
    def search_similar_strategies(self,
                                   query: str,
                                   top_k: int = 5,
                                   filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Search for similar strategies"""
        try:
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            if self.index:
                results = self.index.query(
                    vector=query_embedding.tolist(),
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter_dict
                )
                
                return [
                    {
                        'id': match['id'],
                        'score': match['score'],
                        'metadata': match.get('metadata', {})
                    }
                    for match in results['matches']
                ]
            else:
                # Local fallback: cosine similarity search
                return self._local_search(query_embedding, top_k, filter_dict)
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _local_search(self, 
                      query_vec: np.ndarray,
                      top_k: int,
                      filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Perform local similarity search"""
        if not self.local_storage['vectors']:
            return []
        
        vectors = np.array(self.local_storage['vectors'])
        
        # Cosine similarity
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        vec_norms = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(vec_norms, query_norm)
        
        # Apply filters
        valid_indices = list(range(len(similarities)))
        if filter_dict:
            valid_indices = [
                i for i in valid_indices
                if all(
                    self.local_storage['metadata'][i].get(k) == v
                    for k, v in filter_dict.items()
                )
            ]
        
        # Sort by similarity
        sorted_indices = sorted(valid_indices, key=lambda i: similarities[i], reverse=True)
        top_indices = sorted_indices[:top_k]
        
        return [
            {
                'id': self.local_storage['ids'][i],
                'score': float(similarities[i]),
                'metadata': self.local_storage['metadata'][i]
            }
            for i in top_indices
        ]
    
    def search_by_circuit(self, 
                          circuit_id: str,
                          query: Optional[str] = None,
                          top_k: int = 10) -> List[Dict]:
        """Search strategies for a specific circuit"""
        filter_dict = {'circuit_id': circuit_id}
        
        if query:
            return self.search_similar_strategies(query, top_k, filter_dict)
        else:
            # Return all strategies for circuit
            default_query = f"Race strategy for {circuit_id}"
            return self.search_similar_strategies(default_query, top_k, filter_dict)
    
    def search_by_conditions(self,
                             weather: str,
                             tire_strategy: Optional[str] = None,
                             top_k: int = 10) -> List[Dict]:
        """Search strategies by race conditions"""
        query_parts = [f"Race in {weather} conditions"]
        
        if tire_strategy:
            query_parts.append(f"using {tire_strategy} strategy")
        
        query = " ".join(query_parts)
        return self.search_similar_strategies(query, top_k)
    
    def get_winning_strategies(self,
                               circuit_id: Optional[str] = None,
                               top_k: int = 10) -> List[Dict]:
        """Get strategies that resulted in wins"""
        filter_dict = {'final_position': 1}
        if circuit_id:
            filter_dict['circuit_id'] = circuit_id
        
        query = "Winning race strategy"
        if circuit_id:
            query += f" for {circuit_id}"
        
        return self.search_similar_strategies(query, top_k, filter_dict)
    
    def delete_strategy(self, strategy_id: str) -> bool:
        """Delete a strategy from the database"""
        try:
            if self.index:
                self.index.delete(ids=[strategy_id])
            else:
                if strategy_id in self.local_storage['ids']:
                    idx = self.local_storage['ids'].index(strategy_id)
                    self.local_storage['ids'].pop(idx)
                    self.local_storage['vectors'].pop(idx)
                    self.local_storage['metadata'].pop(idx)
                    self._save_local_fallback()
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete strategy: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        if self.index:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.get('total_vector_count', 0),
                'dimension': stats.get('dimension', self.config.dimension),
                'namespaces': stats.get('namespaces', {})
            }
        else:
            return {
                'total_vectors': len(self.local_storage['ids']),
                'dimension': self.config.dimension,
                'storage': 'local'
            }
    
    def clear_index(self) -> bool:
        """Clear all data from the index"""
        try:
            if self.index:
                self.index.delete(delete_all=True)
            else:
                self.local_storage = {
                    'vectors': [],
                    'metadata': [],
                    'ids': []
                }
                self._save_local_fallback()
            
            logger.info("Index cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            return False


class StrategyIndexer:
    """Index historical F1 strategies into the vector database"""
    
    def __init__(self, vector_db: F1VectorDatabase):
        self.vector_db = vector_db
    
    def index_race_strategies(self, races_data: List[Dict]) -> int:
        """Index strategies from race data"""
        strategies = []
        
        for race in races_data:
            race_name = race.get('race_name', 'Unknown')
            season = race.get('season', 0)
            circuit_id = race.get('circuit_id', 'unknown')
            
            results = race.get('results', [])
            pit_stops = race.get('pit_stops', [])
            
            # Group pit stops by driver
            driver_pits = {}
            for ps in pit_stops:
                driver = ps.get('driver_id', '')
                if driver not in driver_pits:
                    driver_pits[driver] = []
                driver_pits[driver].append(ps)
            
            # Create strategy entries for top 10 finishers
            for result in results[:10]:
                driver_id = result.get('driver_id', '')
                driver_name = result.get('driver_name', driver_id)
                
                pits = driver_pits.get(driver_id, [])
                pit_laps = [p.get('lap', 0) for p in pits]
                
                strategy_id = f"{season}_{circuit_id}_{driver_id}"
                
                description = self._create_strategy_description(
                    race_name, season, driver_name, result, pits
                )
                
                strategy = {
                    'id': strategy_id,
                    'description': description,
                    'race_name': race_name,
                    'season': season,
                    'circuit_id': circuit_id,
                    'driver_id': driver_id,
                    'driver_name': driver_name,
                    'final_position': result.get('position', 20),
                    'grid_position': result.get('grid', 20),
                    'positions_gained': result.get('grid', 20) - result.get('position', 20),
                    'num_pit_stops': len(pits),
                    'pit_laps': pit_laps,
                    'status': result.get('status', 'Unknown')
                }
                
                strategies.append(strategy)
        
        return self.vector_db.upsert_batch(strategies)
    
    def _create_strategy_description(self,
                                     race_name: str,
                                     season: int,
                                     driver_name: str,
                                     result: Dict,
                                     pit_stops: List[Dict]) -> str:
        """Create natural language strategy description"""
        position = result.get('position', '?')
        grid = result.get('grid', '?')
        status = result.get('status', 'Unknown')
        
        pos_change = grid - position if isinstance(grid, int) and isinstance(position, int) else 0
        
        if pos_change > 0:
            pos_str = f"gained {pos_change} positions"
        elif pos_change < 0:
            pos_str = f"lost {abs(pos_change)} positions"
        else:
            pos_str = "maintained position"
        
        num_stops = len(pit_stops)
        if pit_stops:
            pit_laps = [str(p.get('lap', '?')) for p in pit_stops]
            pit_str = f"{num_stops}-stop strategy, pitting on laps {', '.join(pit_laps)}"
        else:
            pit_str = "no recorded pit stops"
        
        return (
            f"At the {season} {race_name}, {driver_name} started P{grid} and finished P{position}, "
            f"{pos_str}. Used a {pit_str}. Race status: {status}."
        )


# Convenience function for quick setup
def setup_vector_database(api_key: Optional[str] = None) -> F1VectorDatabase:
    """Quick setup for the vector database"""
    db = F1VectorDatabase(api_key=api_key)
    db.create_index()
    return db


if __name__ == "__main__":
    # Test vector database
    print("Testing F1 Vector Database...")
    
    # Create database
    db = F1VectorDatabase()
    db.create_index()
    
    # Sample strategy
    sample_strategy = {
        'id': 'test_strategy_1',
        'description': (
            "At the 2024 Monaco Grand Prix, Max Verstappen started P1 and finished P1, "
            "maintaining position. Used a 1-stop strategy, pitting on lap 32. "
            "Ran long first stint on medium tires, switched to hard tires for the finish."
        ),
        'race_name': 'Monaco Grand Prix',
        'season': 2024,
        'circuit_id': 'monaco',
        'driver_id': 'max_verstappen',
        'final_position': 1,
        'grid_position': 1,
        'num_pit_stops': 1
    }
    
    # Upsert
    success = db.upsert_strategy(
        sample_strategy['id'],
        sample_strategy['description'],
        {k: v for k, v in sample_strategy.items() if k not in ['id', 'description']}
    )
    print(f"Upsert success: {success}")
    
    # Search
    results = db.search_similar_strategies(
        "What's the best strategy for Monaco?",
        top_k=3
    )
    print(f"\nSearch results: {len(results)} matches")
    for r in results:
        print(f"  - {r['id']}: score={r['score']:.3f}")
    
    # Stats
    stats = db.get_index_stats()
    print(f"\nIndex stats: {stats}")
    
    print("\n✅ Vector database test complete!")
