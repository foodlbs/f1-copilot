"""
F1 TensorFlow Models Module

Contains machine learning models for F1 race analysis:
- RaceOutcomePredictor: LSTM-based model for predicting race positions
- PitStopOptimizer: Deep Q-Network for pit stop strategy optimization
- TireDegradationModel: Regression model for lap time prediction

Author: F1 Race Strategy Analyzer
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy
from collections import deque
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


@dataclass
class ModelConfig:
    """Configuration for model training"""
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    validation_split: float = 0.2


class RaceOutcomePredictor:
    """
    LSTM-based model for predicting race finishing positions.
    
    Uses sequential lap data to predict final race position.
    Input: Sequence of lap features (position, times, tire state, etc.)
    Output: Predicted finishing position (1-20)
    """
    
    def __init__(self, sequence_length: int = 10, num_features: int = 30):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model = None
        self.history = None
        
    def build_model(self, 
                    lstm_units: List[int] = [128, 64],
                    dense_units: List[int] = [64, 32],
                    dropout_rate: float = 0.3) -> Model:
        """Build the LSTM model architecture"""
        
        inputs = layers.Input(shape=(self.sequence_length, self.num_features))
        
        # LSTM layers
        x = inputs
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            x = layers.LSTM(
                units, 
                return_sequences=return_sequences,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate / 2,
                name=f'lstm_{i+1}'
            )(x)
            x = layers.BatchNormalization()(x)
        
        # Dense layers
        for i, units in enumerate(dense_units):
            x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.BatchNormalization()(x)
        
        # Output layer - position prediction (1-20 normalized to 0-1)
        position_output = layers.Dense(1, activation='sigmoid', name='position')(x)
        
        # Additional output - probability distribution over positions
        prob_output = layers.Dense(20, activation='softmax', name='position_probs')(x)
        
        self.model = Model(inputs=inputs, outputs=[position_output, prob_output])
        
        logger.info(f"Built LSTM model with {self.model.count_params()} parameters")
        return self.model
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compile the model with optimizer and loss functions"""
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={
                'position': 'mse',
                'position_probs': 'categorical_crossentropy'
            },
            loss_weights={
                'position': 1.0,
                'position_probs': 0.5
            },
            metrics={
                'position': ['mae'],
                'position_probs': ['accuracy']
            }
        )
        
        logger.info("Model compiled successfully")
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              config: Optional[ModelConfig] = None) -> Dict:
        """Train the model"""
        config = config or ModelConfig()
        
        if self.model is None:
            self.build_model()
            self.compile_model(config.learning_rate)
        
        # Convert targets to both formats
        y_position = y_train
        y_probs = self._position_to_probs(y_train)
        
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (
                X_val, 
                {'position': y_val, 'position_probs': self._position_to_probs(y_val)}
            )
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=config.early_stopping_patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=config.reduce_lr_patience,
                min_lr=1e-6
            ),
            callbacks.ModelCheckpoint(
                'models/race_predictor_best.keras',
                monitor='val_loss' if validation_data else 'loss',
                save_best_only=True
            )
        ]
        
        self.history = self.model.fit(
            X_train,
            {'position': y_position, 'position_probs': y_probs},
            batch_size=config.batch_size,
            epochs=config.epochs,
            validation_data=validation_data,
            callbacks=callback_list,
            verbose=1
        )
        
        return self.history.history
    
    def _position_to_probs(self, positions: np.ndarray) -> np.ndarray:
        """Convert normalized positions to probability distribution"""
        # positions are 0-1, convert to 0-19 index
        indices = np.clip((positions * 19).astype(int), 0, 19)
        probs = np.zeros((len(positions), 20))
        for i, idx in enumerate(indices):
            probs[i, idx] = 1.0
        return probs
    
    def predict_position(self, sequence: np.ndarray) -> Tuple[int, np.ndarray]:
        """Predict finishing position for a race sequence"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if sequence.ndim == 2:
            sequence = np.expand_dims(sequence, 0)
        
        position_pred, prob_pred = self.model.predict(sequence, verbose=0)
        
        # Convert normalized position back to 1-20 range
        predicted_position = int(np.round(position_pred[0, 0] * 19)) + 1
        probabilities = prob_pred[0]
        
        return predicted_position, probabilities
    
    def save_model(self, path: str):
        """Save the model to disk"""
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a model from disk"""
        self.model = keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")


class PitStopOptimizer:
    """
    Deep Q-Network (DQN) for optimal pit stop strategy.
    
    Uses reinforcement learning to learn when to pit based on
    race state (tire age, position, weather, etc.)
    
    Actions:
    - 0: Stay out
    - 1: Pit for soft tires
    - 2: Pit for medium tires
    - 3: Pit for hard tires
    """
    
    def __init__(self, 
                 state_dim: int = 15,
                 action_dim: int = 4,
                 memory_size: int = 10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        
        # Networks
        self.model = self._build_network()
        self.target_model = self._build_network()
        self.update_target_network()
        
    def _build_network(self) -> Model:
        """Build the Q-network"""
        inputs = layers.Input(shape=(self.state_dim,))
        
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        
        # Dueling DQN architecture
        # Value stream
        value = layers.Dense(32, activation='relu')(x)
        value = layers.Dense(1)(value)
        
        # Advantage stream
        advantage = layers.Dense(32, activation='relu')(x)
        advantage = layers.Dense(self.action_dim)(advantage)
        
        # Combine value and advantage
        q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        
        model = Model(inputs=inputs, outputs=q_values)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def update_target_network(self):
        """Update target network weights"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state: np.ndarray, action: int, 
                 reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state = np.expand_dims(state, 0)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self) -> float:
        """Train on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Double DQN: use main network to select action, target network to evaluate
        next_q_main = self.model.predict(next_states, verbose=0)
        next_actions = np.argmax(next_q_main, axis=1)
        
        next_q_target = self.target_model.predict(next_states, verbose=0)
        next_q_values = next_q_target[np.arange(self.batch_size), next_actions]
        
        # Calculate target Q-values
        targets = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Get current Q-values and update only for taken actions
        current_q = self.model.predict(states, verbose=0)
        current_q[np.arange(self.batch_size), actions] = targets
        
        # Train
        loss = self.model.train_on_batch(states, current_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def calculate_reward(self, 
                         action: int,
                         state: Dict[str, Any],
                         next_state: Dict[str, Any]) -> float:
        """Calculate reward for a pit stop decision"""
        reward = 0.0
        
        # Reward for position improvement
        pos_change = state.get('position', 10) - next_state.get('position', 10)
        reward += pos_change * 2.0
        
        # Penalty for unnecessary pit stops
        if action > 0:  # Pit stop
            reward -= 0.5  # Base pit penalty
            
            tire_age = state.get('tire_age', 0)
            if tire_age < 10:
                reward -= 1.0  # Pitting too early
            elif tire_age > 25:
                reward += 0.5  # Good timing
        
        # Penalty for staying out on old tires
        if action == 0 and state.get('tire_age', 0) > 30:
            reward -= 0.5
        
        # Reward for completing race
        if next_state.get('race_completed', False):
            final_position = next_state.get('position', 20)
            reward += (21 - final_position) * 0.5  # More points for better position
        
        return reward
    
    def get_optimal_strategy(self, 
                             race_state: np.ndarray,
                             remaining_laps: int) -> Dict[str, Any]:
        """Get optimal pit strategy for current race state"""
        action = self.choose_action(race_state, training=False)
        q_values = self.model.predict(np.expand_dims(race_state, 0), verbose=0)[0]
        
        action_names = ['Stay out', 'Pit - Soft', 'Pit - Medium', 'Pit - Hard']
        
        strategy = {
            'recommended_action': action_names[action],
            'action_code': action,
            'confidence': float(np.max(q_values) / np.sum(np.abs(q_values) + 1e-6)),
            'q_values': {name: float(q) for name, q in zip(action_names, q_values)},
            'remaining_laps': remaining_laps
        }
        
        # Add reasoning
        if action == 0:
            strategy['reasoning'] = "Current tire performance is acceptable, no pit stop needed."
        else:
            compound = action_names[action].split(' - ')[1]
            strategy['reasoning'] = f"Recommend pitting for {compound} tires to optimize race performance."
        
        return strategy
    
    def save_model(self, path: str):
        """Save the DQN model"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        self.model.save(f"{path}_main.keras")
        self.target_model.save(f"{path}_target.keras")
        
        # Save training state
        state = {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory)
        }
        with open(f"{path}_state.json", 'w') as f:
            json.dump(state, f)
        
        logger.info(f"DQN model saved to {path}")
    
    def load_model(self, path: str):
        """Load the DQN model"""
        self.model = keras.models.load_model(f"{path}_main.keras")
        self.target_model = keras.models.load_model(f"{path}_target.keras")
        
        if os.path.exists(f"{path}_state.json"):
            with open(f"{path}_state.json", 'r') as f:
                state = json.load(f)
                self.epsilon = state.get('epsilon', 0.01)
        
        logger.info(f"DQN model loaded from {path}")


class TireDegradationModel:
    """
    Regression model for predicting tire degradation and lap times.
    
    Predicts expected lap time based on:
    - Tire compound
    - Tire age (laps)
    - Fuel load
    - Track/weather conditions
    """
    
    def __init__(self, input_dim: int = 8):
        self.input_dim = input_dim
        self.model = None
        self.history = None
        
    def build_model(self) -> Model:
        """Build the degradation prediction model"""
        inputs = layers.Input(shape=(self.input_dim,))
        
        # Main branch
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(32, activation='relu')(x)
        
        # Output: predicted lap time (normalized)
        lap_time = layers.Dense(1, activation='linear', name='lap_time')(x)
        
        # Output: degradation rate (seconds per lap)
        degradation = layers.Dense(1, activation='softplus', name='degradation')(x)
        
        self.model = Model(inputs=inputs, outputs=[lap_time, degradation])
        
        logger.info(f"Built degradation model with {self.model.count_params()} parameters")
        return self.model
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compile the model"""
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={
                'lap_time': 'mse',
                'degradation': 'mse'
            },
            metrics=['mae']
        )
    
    def train(self,
              X_train: np.ndarray,
              y_lap_time: np.ndarray,
              y_degradation: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val_time: Optional[np.ndarray] = None,
              y_val_deg: Optional[np.ndarray] = None,
              config: Optional[ModelConfig] = None) -> Dict:
        """Train the model"""
        config = config or ModelConfig()
        
        if self.model is None:
            self.build_model()
            self.compile_model(config.learning_rate)
        
        validation_data = None
        if X_val is not None:
            validation_data = (
                X_val,
                {'lap_time': y_val_time, 'degradation': y_val_deg}
            )
        
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=config.early_stopping_patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=config.reduce_lr_patience,
                min_lr=1e-6
            )
        ]
        
        self.history = self.model.fit(
            X_train,
            {'lap_time': y_lap_time, 'degradation': y_degradation},
            batch_size=config.batch_size,
            epochs=config.epochs,
            validation_data=validation_data,
            callbacks=callback_list,
            verbose=1
        )
        
        return self.history.history
    
    def predict_stint(self,
                      initial_features: np.ndarray,
                      num_laps: int) -> Dict[str, np.ndarray]:
        """Predict lap times for an entire stint"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        lap_times = []
        degradations = []
        features = initial_features.copy()
        
        for lap in range(num_laps):
            # Update tire age in features (assuming first feature is lap_in_stint)
            features[0] = lap / 30.0  # Normalized tire age
            
            pred = self.model.predict(np.expand_dims(features, 0), verbose=0)
            lap_time, deg = pred[0][0, 0], pred[1][0, 0]
            
            lap_times.append(lap_time)
            degradations.append(deg)
        
        return {
            'lap_times': np.array(lap_times),
            'degradation_rates': np.array(degradations),
            'total_time': np.sum(lap_times),
            'avg_degradation': np.mean(degradations)
        }
    
    def compare_compounds(self,
                          base_features: np.ndarray,
                          stint_length: int) -> Dict[str, Dict]:
        """Compare different tire compounds for a stint"""
        compounds = {
            'SOFT': {'grip': 1.0, 'durability': 0.3},
            'MEDIUM': {'grip': 0.7, 'durability': 0.6},
            'HARD': {'grip': 0.5, 'durability': 0.9}
        }
        
        results = {}
        
        for compound, props in compounds.items():
            features = base_features.copy()
            # Assuming features[4] and [5] are grip and durability
            if len(features) > 5:
                features[4] = props['grip']
                features[5] = props['durability']
            
            stint_pred = self.predict_stint(features, stint_length)
            
            results[compound] = {
                'total_time': float(stint_pred['total_time']),
                'avg_lap_time': float(np.mean(stint_pred['lap_times'])),
                'avg_degradation': float(stint_pred['avg_degradation']),
                'recommended_stint_length': self._calculate_optimal_stint(stint_pred)
            }
        
        return results
    
    def _calculate_optimal_stint(self, stint_pred: Dict) -> int:
        """Calculate optimal stint length based on degradation"""
        lap_times = stint_pred['lap_times']
        
        # Find where degradation causes significant time loss
        if len(lap_times) < 5:
            return len(lap_times)
        
        # Look for when lap times increase significantly
        for i in range(5, len(lap_times)):
            recent_avg = np.mean(lap_times[i-3:i])
            if lap_times[i] > recent_avg * 1.02:  # 2% slower
                return i
        
        return len(lap_times)
    
    def save_model(self, path: str):
        """Save the model"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        self.model.save(path)
        logger.info(f"Degradation model saved to {path}")
    
    def load_model(self, path: str):
        """Load the model"""
        self.model = keras.models.load_model(path)
        logger.info(f"Degradation model loaded from {path}")


class EnsemblePredictor:
    """Ensemble model combining all predictors"""
    
    def __init__(self):
        self.race_predictor = None
        self.pit_optimizer = None
        self.degradation_model = None
        
    def load_all_models(self, model_dir: str = './models'):
        """Load all trained models"""
        try:
            self.race_predictor = RaceOutcomePredictor()
            self.race_predictor.load_model(f"{model_dir}/race_predictor.keras")
        except Exception as e:
            logger.warning(f"Could not load race predictor: {e}")
        
        try:
            self.pit_optimizer = PitStopOptimizer()
            self.pit_optimizer.load_model(f"{model_dir}/pit_optimizer")
        except Exception as e:
            logger.warning(f"Could not load pit optimizer: {e}")
        
        try:
            self.degradation_model = TireDegradationModel()
            self.degradation_model.load_model(f"{model_dir}/degradation_model.keras")
        except Exception as e:
            logger.warning(f"Could not load degradation model: {e}")
    
    def predict_race_outcome(self, 
                             race_sequence: np.ndarray,
                             current_state: np.ndarray,
                             tire_features: np.ndarray,
                             remaining_laps: int) -> Dict[str, Any]:
        """Generate comprehensive race predictions"""
        predictions = {}
        
        # Race position prediction
        if self.race_predictor:
            position, probs = self.race_predictor.predict_position(race_sequence)
            predictions['predicted_position'] = position
            predictions['position_probabilities'] = probs.tolist()
            predictions['top_3_probability'] = float(probs[:3].sum())
        
        # Pit stop recommendation
        if self.pit_optimizer:
            pit_strategy = self.pit_optimizer.get_optimal_strategy(
                current_state, remaining_laps
            )
            predictions['pit_recommendation'] = pit_strategy
        
        # Tire degradation forecast
        if self.degradation_model:
            degradation = self.degradation_model.compare_compounds(
                tire_features, remaining_laps
            )
            predictions['tire_analysis'] = degradation
        
        return predictions


def create_sample_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Create sample training data for testing"""
    sequence_length = 10
    num_features = 30
    
    X = np.random.randn(n_samples, sequence_length, num_features)
    y = np.random.rand(n_samples)  # Normalized positions (0-1)
    
    return X, y


if __name__ == "__main__":
    # Test model creation
    print("Testing TensorFlow Models...")
    
    # Test Race Predictor
    print("\n1. Race Outcome Predictor:")
    predictor = RaceOutcomePredictor(sequence_length=10, num_features=30)
    predictor.build_model()
    predictor.compile_model()
    print(f"   Model built: {predictor.model.count_params()} parameters")
    
    # Test prediction
    sample_sequence = np.random.randn(1, 10, 30)
    position, probs = predictor.predict_position(sample_sequence)
    print(f"   Sample prediction: P{position}")
    
    # Test Pit Stop Optimizer
    print("\n2. Pit Stop Optimizer:")
    optimizer = PitStopOptimizer(state_dim=15, action_dim=4)
    print(f"   DQN built: {optimizer.model.count_params()} parameters")
    
    sample_state = np.random.randn(15)
    strategy = optimizer.get_optimal_strategy(sample_state, remaining_laps=30)
    print(f"   Sample strategy: {strategy['recommended_action']}")
    
    # Test Tire Degradation Model
    print("\n3. Tire Degradation Model:")
    degradation = TireDegradationModel(input_dim=8)
    degradation.build_model()
    degradation.compile_model()
    print(f"   Model built: {degradation.model.count_params()} parameters")
    
    print("\n✅ All models created successfully!")
