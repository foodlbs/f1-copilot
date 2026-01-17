"""
F1 Model Training Script

Trains all ML models for the F1 Race Strategy Analyzer:
1. Race Outcome Predictor (LSTM)
2. Pit Stop Optimizer (DQN)
3. Tire Degradation Model

Author: F1 Race Strategy Analyzer
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories"""
    dirs = ['./models', './data/raw', './data/processed', './cache', './logs']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info("Directories created")


def collect_training_data(start_year: int = 2015, 
                          end_year: int = 2024,
                          force_refresh: bool = False) -> List[Dict]:
    """Collect historical F1 data for training"""
    logger.info(f"Collecting data from {start_year} to {end_year}...")
    
    data_file = f"./data/raw/seasons_{start_year}_{end_year}.json"
    
    if os.path.exists(data_file) and not force_refresh:
        logger.info(f"Loading existing data from {data_file}")
        with open(data_file, 'r') as f:
            return json.load(f)
    
    from data_collection import F1DataCollector
    
    collector = F1DataCollector(use_fastf1=False)
    seasons_data = collector.get_seasons_data(start_year, end_year)
    
    # Save data
    with open(data_file, 'w') as f:
        json.dump(seasons_data, f, default=str)
    
    logger.info(f"Collected and saved data for {len(seasons_data)} seasons")
    return seasons_data


def prepare_features(seasons_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Prepare features from collected data"""
    logger.info("Preparing features...")
    
    from feature_engineering import F1FeatureEngineer, FeatureConfig
    
    config = FeatureConfig(
        sequence_length=10,
        include_weather=True,
        normalize=True
    )
    
    engineer = F1FeatureEngineer(config)
    
    # Flatten races from all seasons
    all_races = []
    for season in seasons_data:
        all_races.extend(season.get('races', []))
    
    logger.info(f"Processing {len(all_races)} races...")
    
    # Prepare training data
    X, y, metadata = engineer.prepare_training_data(all_races)
    
    logger.info(f"Features prepared: X shape = {X.shape}, y shape = {y.shape}")
    
    # Save feature engineer for later use
    engineer.save_scalers('./models/scalers.joblib')
    
    return X, y, metadata


def train_race_predictor(X: np.ndarray, y: np.ndarray,
                         epochs: int = 100,
                         batch_size: int = 32) -> Dict:
    """Train the race outcome prediction model"""
    logger.info("Training Race Outcome Predictor...")
    
    from tensorflow_models import RaceOutcomePredictor, ModelConfig
    from feature_engineering import F1FeatureEngineer
    
    # Get feature dimensions
    sequence_length = X.shape[1]
    num_features = X.shape[2]
    
    # Create model
    predictor = RaceOutcomePredictor(
        sequence_length=sequence_length,
        num_features=num_features
    )
    
    # Build and compile
    predictor.build_model(
        lstm_units=[128, 64],
        dense_units=[64, 32],
        dropout_rate=0.3
    )
    predictor.compile_model(learning_rate=0.001)
    
    # Split data
    engineer = F1FeatureEngineer()
    X_train, X_val, X_test, y_train, y_val, y_test = engineer.split_data(X, y)
    
    logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Train
    config = ModelConfig(
        batch_size=batch_size,
        epochs=epochs,
        early_stopping_patience=15,
        reduce_lr_patience=5
    )
    
    history = predictor.train(X_train, y_train, X_val, y_val, config)
    
    # Evaluate on test set
    test_loss = predictor.model.evaluate(
        X_test, 
        {'position': y_test, 'position_probs': predictor._position_to_probs(y_test)},
        verbose=0
    )
    
    # Save model
    predictor.save_model('./models/race_predictor.keras')
    
    results = {
        'model': 'RaceOutcomePredictor',
        'train_samples': X_train.shape[0],
        'test_loss': float(test_loss[0]) if isinstance(test_loss, list) else float(test_loss),
        'epochs_trained': len(history.get('loss', [])),
        'final_train_loss': history.get('loss', [0])[-1],
        'final_val_loss': history.get('val_loss', [0])[-1] if 'val_loss' in history else None
    }
    
    logger.info(f"Race Predictor trained: Test Loss = {results['test_loss']:.4f}")
    
    return results


def train_pit_stop_optimizer(seasons_data: List[Dict],
                              episodes: int = 1000) -> Dict:
    """Train the pit stop optimizer using simulated races"""
    logger.info("Training Pit Stop Optimizer...")
    
    from tensorflow_models import PitStopOptimizer
    from feature_engineering import F1FeatureEngineer
    
    # Create optimizer
    optimizer = PitStopOptimizer(state_dim=15, action_dim=4)
    engineer = F1FeatureEngineer()
    
    # Simulate training episodes
    total_rewards = []
    
    for episode in range(episodes):
        # Simulate a race
        state = simulate_race_state()
        episode_reward = 0
        done = False
        
        while not done:
            # Choose action
            action = optimizer.choose_action(state)
            
            # Simulate next state
            next_state, reward, done = simulate_race_step(state, action)
            
            # Store experience
            optimizer.remember(state, action, reward, next_state, done)
            
            # Train
            optimizer.replay()
            
            episode_reward += reward
            state = next_state
        
        total_rewards.append(episode_reward)
        
        # Update target network periodically
        if episode % 10 == 0:
            optimizer.update_target_network()
        
        if episode % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            logger.info(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Epsilon = {optimizer.epsilon:.3f}")
    
    # Save model
    optimizer.save_model('./models/pit_optimizer')
    
    results = {
        'model': 'PitStopOptimizer',
        'episodes': episodes,
        'final_avg_reward': float(np.mean(total_rewards[-100:])),
        'final_epsilon': optimizer.epsilon
    }
    
    logger.info(f"Pit Stop Optimizer trained: Final Avg Reward = {results['final_avg_reward']:.2f}")
    
    return results


def simulate_race_state() -> np.ndarray:
    """Simulate a race state for DQN training"""
    return np.array([
        np.random.uniform(0, 1),      # Race progress
        np.random.uniform(0, 1),      # Remaining laps (normalized)
        np.random.uniform(0, 1),      # Position (normalized)
        np.random.uniform(0, 1),      # Tire age (normalized)
        np.random.uniform(-0.1, 0.1), # Lap time trend
        np.random.uniform(0, 1),      # Pit stops done
        np.random.choice([0, 1]),     # High tire age flag
        np.random.uniform(0.3, 0.7),  # Circuit overtaking difficulty
        np.random.uniform(0.3, 0.7),  # Tire wear factor
        np.random.uniform(0.2, 0.5),  # Safety car probability
        np.random.uniform(0.5, 1.0),  # Current tire grip
        np.random.uniform(0.3, 0.9),  # Current tire durability
        np.random.uniform(0, 1),      # Track temp
        np.random.uniform(0, 1),      # Weather condition
        np.random.uniform(0, 1)       # Gap to car ahead
    ])


def simulate_race_step(state: np.ndarray, 
                       action: int) -> Tuple[np.ndarray, float, bool]:
    """Simulate one step of a race"""
    next_state = state.copy()
    reward = 0.0
    
    # Update race progress
    progress_increment = 1.0 / 50  # Assume 50-lap race
    next_state[0] = min(1.0, state[0] + progress_increment)
    next_state[1] = max(0.0, state[1] - progress_increment)
    
    # Update tire age
    tire_age = state[3]
    
    if action == 0:  # Stay out
        next_state[3] = min(1.0, tire_age + 0.033)  # Age increases
        
        # Penalty for very old tires
        if tire_age > 0.8:
            reward -= 0.5
            # May lose position
            if np.random.random() < 0.3:
                next_state[2] = min(1.0, next_state[2] + 0.05)
    else:  # Pit stop
        next_state[3] = 0.0  # Reset tire age
        next_state[5] = min(1.0, state[5] + 0.33)  # Increment stops
        
        # Pit stop costs time (position loss)
        reward -= 0.3
        next_state[2] = min(1.0, next_state[2] + 0.1)
        
        # Update tire characteristics based on compound choice
        if action == 1:  # Soft
            next_state[10] = 1.0
            next_state[11] = 0.3
        elif action == 2:  # Medium
            next_state[10] = 0.7
            next_state[11] = 0.6
        else:  # Hard
            next_state[10] = 0.5
            next_state[11] = 0.9
    
    # Simulate position changes based on tire state and random events
    tire_performance = next_state[10] * (1 - next_state[3] * 0.5)
    if tire_performance > 0.7 and np.random.random() < 0.2:
        # Good tires, chance to overtake
        next_state[2] = max(0.0, next_state[2] - 0.05)
        reward += 0.2
    
    # Check if race is done
    done = next_state[0] >= 1.0
    
    if done:
        # Final position reward
        final_position = int(next_state[2] * 19) + 1
        reward += (21 - final_position) * 0.5
    
    return next_state, reward, done


def train_tire_degradation_model(seasons_data: List[Dict],
                                  epochs: int = 100) -> Dict:
    """Train the tire degradation prediction model"""
    logger.info("Training Tire Degradation Model...")
    
    from tensorflow_models import TireDegradationModel, ModelConfig
    
    # Generate synthetic training data (in production, would use real data)
    n_samples = 5000
    
    # Features: lap_in_stint, fuel_load, track_temp, air_temp, grip, durability, speed, cornering
    X = np.random.rand(n_samples, 8)
    
    # Targets: lap_time (normalized), degradation_rate
    base_lap_time = 1.5  # minutes, normalized
    y_lap_time = base_lap_time + X[:, 0] * 0.1 * (1 - X[:, 5])  # Slower with age, less with durability
    y_lap_time += np.random.normal(0, 0.02, n_samples)
    
    y_degradation = 0.02 + 0.08 * (1 - X[:, 5]) + 0.05 * X[:, 2]  # Depends on durability and track temp
    y_degradation = np.clip(y_degradation + np.random.normal(0, 0.01, n_samples), 0, 0.2)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_time_train, y_time_test, y_deg_train, y_deg_test = train_test_split(
        X, y_lap_time, y_degradation, test_size=0.2, random_state=42
    )
    
    # Further split for validation
    X_train, X_val, y_time_train, y_time_val, y_deg_train, y_deg_val = train_test_split(
        X_train, y_time_train, y_deg_train, test_size=0.15, random_state=42
    )
    
    # Create and train model
    model = TireDegradationModel(input_dim=8)
    model.build_model()
    model.compile_model(learning_rate=0.001)
    
    config = ModelConfig(
        batch_size=32,
        epochs=epochs,
        early_stopping_patience=10
    )
    
    history = model.train(
        X_train, y_time_train, y_deg_train,
        X_val, y_time_val, y_deg_val,
        config
    )
    
    # Evaluate
    test_results = model.model.evaluate(
        X_test, 
        {'lap_time': y_time_test, 'degradation': y_deg_test},
        verbose=0
    )
    
    # Save model
    model.save_model('./models/degradation_model.keras')
    
    results = {
        'model': 'TireDegradationModel',
        'train_samples': X_train.shape[0],
        'test_loss': float(test_results[0]),
        'epochs_trained': len(history.get('loss', [])),
    }
    
    logger.info(f"Tire Degradation Model trained: Test Loss = {results['test_loss']:.4f}")
    
    return results


def initialize_vector_database(seasons_data: List[Dict]) -> Dict:
    """Initialize and populate the vector database"""
    logger.info("Initializing Vector Database...")
    
    from vector_database import F1VectorDatabase, StrategyIndexer
    
    db = F1VectorDatabase()
    success = db.create_index()
    
    if not success:
        logger.warning("Vector database initialization failed")
        return {'status': 'failed'}
    
    # Index strategies
    all_races = []
    for season in seasons_data:
        all_races.extend(season.get('races', []))
    
    indexer = StrategyIndexer(db)
    count = indexer.index_race_strategies(all_races)
    
    stats = db.get_index_stats()
    
    results = {
        'status': 'success',
        'strategies_indexed': count,
        'total_vectors': stats.get('total_vectors', 0)
    }
    
    logger.info(f"Vector database initialized: {count} strategies indexed")
    
    return results


def run_training_pipeline(args):
    """Run the complete training pipeline"""
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("F1 Race Strategy Analyzer - Training Pipeline")
    logger.info("=" * 60)
    
    # Setup
    setup_directories()
    
    results = {
        'start_time': start_time.isoformat(),
        'models': {}
    }
    
    try:
        # Collect data
        if not args.skip_collection:
            logger.info("\n[1/5] Collecting training data...")
            seasons_data = collect_training_data(
                args.start_year,
                args.end_year,
                args.force_refresh
            )
        else:
            logger.info("\n[1/5] Loading existing data...")
            data_file = f"./data/raw/seasons_{args.start_year}_{args.end_year}.json"
            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    seasons_data = json.load(f)
            else:
                # Use minimal sample data
                logger.warning("No existing data found, using sample data")
                seasons_data = [{'season': 2024, 'races': []}]
        
        # Prepare features
        logger.info("\n[2/5] Preparing features...")
        X, y, metadata = prepare_features(seasons_data)
        
        # Train Race Predictor
        if not args.skip_lstm:
            logger.info("\n[3/5] Training Race Outcome Predictor...")
            lstm_results = train_race_predictor(X, y, args.epochs, args.batch_size)
            results['models']['race_predictor'] = lstm_results
        else:
            logger.info("\n[3/5] Skipping Race Outcome Predictor")
        
        # Train Pit Stop Optimizer
        if not args.skip_dqn:
            logger.info("\n[4/5] Training Pit Stop Optimizer...")
            dqn_results = train_pit_stop_optimizer(seasons_data, args.dqn_episodes)
            results['models']['pit_optimizer'] = dqn_results
        else:
            logger.info("\n[4/5] Skipping Pit Stop Optimizer")
        
        # Train Tire Degradation Model
        if not args.skip_degradation:
            logger.info("\n[5/5] Training Tire Degradation Model...")
            deg_results = train_tire_degradation_model(seasons_data, args.epochs)
            results['models']['degradation_model'] = deg_results
        else:
            logger.info("\n[5/5] Skipping Tire Degradation Model")
        
        # Initialize vector database
        if args.init_vectordb:
            logger.info("\n[Bonus] Initializing Vector Database...")
            vdb_results = initialize_vector_database(seasons_data)
            results['vector_database'] = vdb_results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        results['error'] = str(e)
        raise
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    results['end_time'] = end_time.isoformat()
    results['duration_seconds'] = duration
    
    # Save results
    results_file = f"./logs/training_results_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"Results saved to: {results_file}")
    logger.info("\nModels saved:")
    logger.info("  - ./models/race_predictor.keras")
    logger.info("  - ./models/pit_optimizer_main.keras")
    logger.info("  - ./models/degradation_model.keras")
    logger.info("  - ./models/scalers.joblib")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train F1 Race Strategy Analyzer models'
    )
    
    # Data collection arguments
    parser.add_argument('--start-year', type=int, default=2020,
                        help='Start year for data collection (default: 2020)')
    parser.add_argument('--end-year', type=int, default=2024,
                        help='End year for data collection (default: 2024)')
    parser.add_argument('--force-refresh', action='store_true',
                        help='Force refresh of cached data')
    parser.add_argument('--skip-collection', action='store_true',
                        help='Skip data collection, use existing data')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size (default: 32)')
    parser.add_argument('--dqn-episodes', type=int, default=500,
                        help='Number of DQN training episodes (default: 500)')
    
    # Skip arguments
    parser.add_argument('--skip-lstm', action='store_true',
                        help='Skip LSTM race predictor training')
    parser.add_argument('--skip-dqn', action='store_true',
                        help='Skip DQN pit optimizer training')
    parser.add_argument('--skip-degradation', action='store_true',
                        help='Skip tire degradation model training')
    parser.add_argument('--init-vectordb', action='store_true',
                        help='Initialize and populate vector database')
    
    # Quick mode
    parser.add_argument('--quick', action='store_true',
                        help='Quick training with reduced parameters')
    
    args = parser.parse_args()
    
    # Apply quick mode settings
    if args.quick:
        args.start_year = 2023
        args.end_year = 2024
        args.epochs = 10
        args.dqn_episodes = 100
        logger.info("Quick mode enabled: reduced training parameters")
    
    # Run training
    try:
        results = run_training_pipeline(args)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
