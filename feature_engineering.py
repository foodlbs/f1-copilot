"""
F1 Feature Engineering Module

Transforms raw F1 data into features suitable for machine learning models.
Handles race data, telemetry, weather, and historical performance features.

Author: F1 Race Strategy Analyzer
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    sequence_length: int = 10  # Number of laps to use as sequence
    include_weather: bool = True
    include_telemetry: bool = False
    normalize: bool = True
    encode_categorical: bool = True


class CircuitEncoder:
    """Encodes circuit characteristics into numerical features"""
    
    # Circuit characteristics database
    CIRCUIT_FEATURES = {
        'monaco': {
            'circuit_type': 'street',
            'length_km': 3.337,
            'corners': 19,
            'drs_zones': 1,
            'overtaking_difficulty': 0.95,
            'tire_wear': 0.3,
            'fuel_consumption': 0.7,
            'safety_car_probability': 0.6,
            'avg_pit_time_loss': 25.0
        },
        'monza': {
            'circuit_type': 'power',
            'length_km': 5.793,
            'corners': 11,
            'drs_zones': 2,
            'overtaking_difficulty': 0.3,
            'tire_wear': 0.4,
            'fuel_consumption': 0.9,
            'safety_car_probability': 0.3,
            'avg_pit_time_loss': 22.0
        },
        'silverstone': {
            'circuit_type': 'balanced',
            'length_km': 5.891,
            'corners': 18,
            'drs_zones': 2,
            'overtaking_difficulty': 0.4,
            'tire_wear': 0.7,
            'fuel_consumption': 0.75,
            'safety_car_probability': 0.35,
            'avg_pit_time_loss': 21.0
        },
        'spa': {
            'circuit_type': 'power',
            'length_km': 7.004,
            'corners': 19,
            'drs_zones': 2,
            'overtaking_difficulty': 0.35,
            'tire_wear': 0.5,
            'fuel_consumption': 0.85,
            'safety_car_probability': 0.45,
            'avg_pit_time_loss': 23.0
        },
        'suzuka': {
            'circuit_type': 'technical',
            'length_km': 5.807,
            'corners': 18,
            'drs_zones': 1,
            'overtaking_difficulty': 0.55,
            'tire_wear': 0.65,
            'fuel_consumption': 0.7,
            'safety_car_probability': 0.3,
            'avg_pit_time_loss': 22.0
        },
        'bahrain': {
            'circuit_type': 'balanced',
            'length_km': 5.412,
            'corners': 15,
            'drs_zones': 3,
            'overtaking_difficulty': 0.35,
            'tire_wear': 0.6,
            'fuel_consumption': 0.7,
            'safety_car_probability': 0.35,
            'avg_pit_time_loss': 21.0
        },
        'jeddah': {
            'circuit_type': 'street',
            'length_km': 6.174,
            'corners': 27,
            'drs_zones': 3,
            'overtaking_difficulty': 0.4,
            'tire_wear': 0.45,
            'fuel_consumption': 0.8,
            'safety_car_probability': 0.55,
            'avg_pit_time_loss': 22.0
        },
        'default': {
            'circuit_type': 'balanced',
            'length_km': 5.0,
            'corners': 15,
            'drs_zones': 2,
            'overtaking_difficulty': 0.5,
            'tire_wear': 0.5,
            'fuel_consumption': 0.75,
            'safety_car_probability': 0.35,
            'avg_pit_time_loss': 22.0
        }
    }
    
    CIRCUIT_TYPE_ENCODING = {
        'street': [1, 0, 0, 0],
        'power': [0, 1, 0, 0],
        'technical': [0, 0, 1, 0],
        'balanced': [0, 0, 0, 1]
    }
    
    def encode(self, circuit_id: str) -> np.ndarray:
        """Encode circuit into feature vector"""
        circuit_id = circuit_id.lower()
        features = self.CIRCUIT_FEATURES.get(circuit_id, self.CIRCUIT_FEATURES['default'])
        
        # Numerical features
        numerical = [
            features['length_km'] / 10.0,  # Normalize
            features['corners'] / 30.0,
            features['drs_zones'] / 4.0,
            features['overtaking_difficulty'],
            features['tire_wear'],
            features['fuel_consumption'],
            features['safety_car_probability'],
            features['avg_pit_time_loss'] / 30.0
        ]
        
        # Circuit type encoding
        type_encoding = self.CIRCUIT_TYPE_ENCODING.get(
            features['circuit_type'], 
            self.CIRCUIT_TYPE_ENCODING['balanced']
        )
        
        return np.array(numerical + type_encoding)


class TireCompoundEncoder:
    """Encodes tire compound information"""
    
    COMPOUND_FEATURES = {
        'SOFT': {'grip': 1.0, 'durability': 0.3, 'optimal_temp': 100},
        'MEDIUM': {'grip': 0.7, 'durability': 0.6, 'optimal_temp': 95},
        'HARD': {'grip': 0.5, 'durability': 0.9, 'optimal_temp': 90},
        'INTERMEDIATE': {'grip': 0.8, 'durability': 0.5, 'optimal_temp': 75},
        'WET': {'grip': 0.6, 'durability': 0.7, 'optimal_temp': 70},
        'UNKNOWN': {'grip': 0.7, 'durability': 0.6, 'optimal_temp': 95}
    }
    
    def encode(self, compound: str) -> np.ndarray:
        """Encode tire compound into feature vector"""
        compound = compound.upper() if compound else 'UNKNOWN'
        features = self.COMPOUND_FEATURES.get(compound, self.COMPOUND_FEATURES['UNKNOWN'])
        
        # One-hot encoding for compound type
        compounds = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
        one_hot = [1 if c == compound else 0 for c in compounds]
        
        # Numerical features
        numerical = [
            features['grip'],
            features['durability'],
            features['optimal_temp'] / 100.0
        ]
        
        return np.array(numerical + one_hot)


class WeatherEncoder:
    """Encodes weather conditions"""
    
    def encode(self, weather: Dict[str, Any]) -> np.ndarray:
        """Encode weather into feature vector"""
        if not weather:
            return np.zeros(6)
        
        return np.array([
            weather.get('track_temp_avg', 30) / 60.0,  # Normalize to 0-1
            weather.get('air_temp_avg', 25) / 40.0,
            weather.get('humidity_avg', 50) / 100.0,
            1.0 if weather.get('rainfall', False) else 0.0,
            weather.get('wind_speed', 0) / 30.0,
            weather.get('wind_direction', 0) / 360.0
        ])


class DriverEncoder:
    """Encodes driver information and performance"""
    
    def __init__(self):
        self.driver_stats = {}
        self.label_encoder = LabelEncoder()
        self.fitted = False
    
    def fit(self, drivers: List[str]):
        """Fit the encoder with known drivers"""
        self.label_encoder.fit(drivers + ['unknown'])
        self.fitted = True
    
    def update_stats(self, driver_id: str, race_data: Dict):
        """Update driver statistics with new race data"""
        if driver_id not in self.driver_stats:
            self.driver_stats[driver_id] = {
                'races': 0,
                'wins': 0,
                'podiums': 0,
                'avg_position': [],
                'avg_qualifying': [],
                'dnfs': 0
            }
        
        stats = self.driver_stats[driver_id]
        stats['races'] += 1
        
        position = race_data.get('position', 20)
        if position == 1:
            stats['wins'] += 1
        if position <= 3:
            stats['podiums'] += 1
        if race_data.get('status', 'Finished') != 'Finished':
            stats['dnfs'] += 1
        
        stats['avg_position'].append(position)
        stats['avg_qualifying'].append(race_data.get('grid', 20))
    
    def encode(self, driver_id: str) -> np.ndarray:
        """Encode driver into feature vector"""
        if not self.fitted:
            # Return default encoding if not fitted
            return np.zeros(10)
        
        try:
            driver_idx = self.label_encoder.transform([driver_id])[0]
        except:
            driver_idx = self.label_encoder.transform(['unknown'])[0]
        
        # Normalize driver index
        num_drivers = len(self.label_encoder.classes_)
        driver_encoding = [driver_idx / num_drivers]
        
        # Add performance stats if available
        if driver_id in self.driver_stats:
            stats = self.driver_stats[driver_id]
            races = max(stats['races'], 1)
            
            performance = [
                stats['wins'] / races,
                stats['podiums'] / races,
                np.mean(stats['avg_position']) / 20 if stats['avg_position'] else 0.5,
                np.mean(stats['avg_qualifying']) / 20 if stats['avg_qualifying'] else 0.5,
                stats['dnfs'] / races,
                min(races / 100, 1.0),  # Experience (capped)
            ]
        else:
            performance = [0.0, 0.0, 0.5, 0.5, 0.0, 0.0]
        
        return np.array(driver_encoding + performance + [0.0, 0.0, 0.0])


class F1FeatureEngineer:
    """Main feature engineering class for F1 data"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        
        # Encoders
        self.circuit_encoder = CircuitEncoder()
        self.tire_encoder = TireCompoundEncoder()
        self.weather_encoder = WeatherEncoder()
        self.driver_encoder = DriverEncoder()
        
        # Scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        
        # Feature dimensions
        self.feature_dim = 30  # Will be updated dynamically
        
    def prepare_race_sequence(self, race_data: Dict, 
                               driver_id: str) -> Tuple[np.ndarray, Dict]:
        """Prepare sequence features for a single race"""
        features = []
        
        # Static features (same for all laps)
        circuit_features = self.circuit_encoder.encode(race_data.get('circuit_id', ''))
        weather_features = self.weather_encoder.encode(race_data.get('weather', {}))
        driver_features = self.driver_encoder.encode(driver_id)
        
        # Get lap times for this driver
        lap_times = [lt for lt in race_data.get('lap_times', []) 
                     if lt.get('driver_id') == driver_id]
        lap_times.sort(key=lambda x: x.get('lap', 0))
        
        # Get pit stops for this driver
        pit_stops = [ps for ps in race_data.get('pit_stops', [])
                     if ps.get('driver_id') == driver_id]
        pit_stop_laps = {ps.get('lap'): ps for ps in pit_stops}
        
        # Create sequence
        total_laps = race_data.get('total_laps', 50)
        current_tire = 'MEDIUM'
        tire_age = 0
        
        for lap_num in range(1, min(len(lap_times) + 1, self.config.sequence_length + 1)):
            lap_data = lap_times[lap_num - 1] if lap_num <= len(lap_times) else {}
            
            # Check for pit stop
            if lap_num in pit_stop_laps:
                current_tire = 'MEDIUM'  # Assume compound change
                tire_age = 0
            else:
                tire_age += 1
            
            # Lap-specific features
            lap_features = np.array([
                lap_num / total_laps,  # Race progress
                lap_data.get('position', 10) / 20,  # Position
                lap_data.get('time_millis', 90000) / 120000,  # Lap time normalized
                tire_age / 30,  # Tire age
                1.0 if lap_num in pit_stop_laps else 0.0,  # Pit stop indicator
                (total_laps - lap_num) / total_laps,  # Remaining laps
            ])
            
            tire_features = self.tire_encoder.encode(current_tire)
            
            # Combine all features for this lap
            combined = np.concatenate([
                lap_features,
                tire_features[:3],  # Just grip/durability/temp
                circuit_features[:8],  # Circuit features
                weather_features[:4],  # Weather features
                driver_features[:6]  # Driver features
            ])
            
            features.append(combined)
        
        # Pad sequence if needed
        while len(features) < self.config.sequence_length:
            features.insert(0, np.zeros_like(features[0]) if features else np.zeros(27))
        
        feature_array = np.array(features[-self.config.sequence_length:])
        
        # Metadata for reference
        metadata = {
            'driver_id': driver_id,
            'race': race_data.get('race_name', ''),
            'season': race_data.get('season', 0),
            'total_laps': total_laps
        }
        
        return feature_array, metadata
    
    def prepare_training_data(self, races_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Prepare training data from multiple races"""
        X = []
        y = []
        metadata = []
        
        for race in races_data:
            results = race.get('results', [])
            
            for result in results:
                driver_id = result.get('driver_id', '')
                
                try:
                    features, meta = self.prepare_race_sequence(race, driver_id)
                    
                    # Target: final position (normalized)
                    position = result.get('position', 20)
                    target = (position - 1) / 19  # 0 = win, 1 = 20th
                    
                    X.append(features)
                    y.append(target)
                    metadata.append({**meta, 'actual_position': position})
                    
                except Exception as e:
                    logger.warning(f"Failed to process {driver_id}: {e}")
                    continue
        
        return np.array(X), np.array(y), metadata
    
    def prepare_pit_stop_features(self, race_data: Dict, 
                                   current_lap: int,
                                   driver_id: str) -> np.ndarray:
        """Prepare features for pit stop decision model"""
        total_laps = race_data.get('total_laps', 50)
        
        # Get driver's current state
        lap_times = [lt for lt in race_data.get('lap_times', [])
                     if lt.get('driver_id') == driver_id]
        
        current_position = 10
        recent_lap_times = []
        
        for lt in lap_times:
            if lt.get('lap', 0) <= current_lap:
                current_position = lt.get('position', current_position)
                recent_lap_times.append(lt.get('time_millis', 90000))
        
        # Calculate lap time trend (degradation indicator)
        if len(recent_lap_times) >= 3:
            recent = recent_lap_times[-3:]
            lap_trend = (recent[-1] - recent[0]) / recent[0]
        else:
            lap_trend = 0
        
        # Get tire age
        pit_stops = [ps for ps in race_data.get('pit_stops', [])
                     if ps.get('driver_id') == driver_id and ps.get('lap', 0) < current_lap]
        last_pit = max([ps.get('lap', 0) for ps in pit_stops]) if pit_stops else 0
        tire_age = current_lap - last_pit
        
        # Circuit and weather features
        circuit_features = self.circuit_encoder.encode(race_data.get('circuit_id', ''))
        weather_features = self.weather_encoder.encode(race_data.get('weather', {}))
        
        features = np.concatenate([
            np.array([
                current_lap / total_laps,  # Race progress
                (total_laps - current_lap) / total_laps,  # Remaining
                current_position / 20,  # Position
                tire_age / 30,  # Tire age
                lap_trend,  # Degradation trend
                len(pit_stops) / 3,  # Stops so far
                1 if tire_age > 20 else 0,  # High tire age flag
            ]),
            circuit_features[:5],
            weather_features[:3]
        ])
        
        return features
    
    def prepare_tire_degradation_features(self, stint_data: Dict) -> np.ndarray:
        """Prepare features for tire degradation prediction"""
        features = np.array([
            stint_data.get('lap_in_stint', 0) / 30,
            stint_data.get('fuel_load', 100) / 110,
            stint_data.get('track_temp', 30) / 60,
            stint_data.get('air_temp', 25) / 40,
            stint_data.get('tire_compound_grip', 0.7),
            stint_data.get('circuit_tire_wear', 0.5),
            stint_data.get('avg_speed', 200) / 250,
            stint_data.get('cornering_intensity', 0.5),
        ])
        
        return features
    
    def create_strategy_embedding_features(self, race_data: Dict,
                                            driver_id: str) -> Dict[str, Any]:
        """Create features for strategy embedding in vector database"""
        results = race_data.get('results', [])
        driver_result = next((r for r in results if r.get('driver_id') == driver_id), {})
        
        pit_stops = [ps for ps in race_data.get('pit_stops', [])
                     if ps.get('driver_id') == driver_id]
        
        # Strategy summary
        strategy_features = {
            'race_name': race_data.get('race_name', ''),
            'circuit_id': race_data.get('circuit_id', ''),
            'season': race_data.get('season', 0),
            'driver_id': driver_id,
            'final_position': driver_result.get('position', 20),
            'grid_position': driver_result.get('grid', 20),
            'positions_gained': driver_result.get('grid', 20) - driver_result.get('position', 20),
            'num_pit_stops': len(pit_stops),
            'pit_stop_laps': [ps.get('lap') for ps in pit_stops],
            'avg_pit_duration': np.mean([ps.get('duration', 25) for ps in pit_stops]) if pit_stops else 0,
            'status': driver_result.get('status', 'Unknown'),
            'weather': race_data.get('weather', {}),
        }
        
        # Create text description for embedding
        description = self._create_strategy_description(strategy_features)
        strategy_features['description'] = description
        
        return strategy_features
    
    def _create_strategy_description(self, features: Dict) -> str:
        """Create natural language description of a strategy"""
        pit_laps = features.get('pit_stop_laps', [])
        pit_str = ', '.join([f"lap {l}" for l in pit_laps]) if pit_laps else 'no stops'
        
        position_change = features.get('positions_gained', 0)
        if position_change > 0:
            pos_str = f"gained {position_change} positions"
        elif position_change < 0:
            pos_str = f"lost {abs(position_change)} positions"
        else:
            pos_str = "maintained position"
        
        description = (
            f"At {features.get('race_name', 'Unknown')} ({features.get('season', '')}), "
            f"driver started P{features.get('grid_position', '?')} and finished P{features.get('final_position', '?')}, "
            f"{pos_str}. "
            f"Strategy: {features.get('num_pit_stops', 0)}-stop ({pit_str}). "
            f"Average pit stop time: {features.get('avg_pit_duration', 0):.1f}s. "
            f"Status: {features.get('status', 'Unknown')}."
        )
        
        return description
    
    def normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features using StandardScaler"""
        original_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        
        if fit:
            X_normalized = self.feature_scaler.fit_transform(X_flat)
        else:
            X_normalized = self.feature_scaler.transform(X_flat)
        
        return X_normalized.reshape(original_shape)
    
    def split_data(self, X: np.ndarray, y: np.ndarray,
                   test_size: float = 0.2,
                   val_size: float = 0.1) -> Tuple[np.ndarray, ...]:
        """Split data into train, validation, and test sets"""
        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_ratio, random_state=42
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_scalers(self, path: str):
        """Save fitted scalers"""
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler
        }, path)
        logger.info(f"Saved scalers to {path}")
    
    def load_scalers(self, path: str):
        """Load fitted scalers"""
        import joblib
        scalers = joblib.load(path)
        self.feature_scaler = scalers['feature_scaler']
        self.target_scaler = scalers['target_scaler']
        logger.info(f"Loaded scalers from {path}")


def create_sample_features() -> Dict[str, np.ndarray]:
    """Create sample features for testing"""
    config = FeatureConfig(sequence_length=10)
    engineer = F1FeatureEngineer(config)
    
    # Sample race data
    sample_race = {
        'season': 2024,
        'round': 1,
        'race_name': 'Bahrain Grand Prix',
        'circuit_id': 'bahrain',
        'total_laps': 57,
        'weather': {
            'track_temp_avg': 35,
            'air_temp_avg': 28,
            'humidity_avg': 45,
            'rainfall': False
        },
        'results': [
            {'driver_id': 'max_verstappen', 'position': 1, 'grid': 1, 'status': 'Finished'},
            {'driver_id': 'sergio_perez', 'position': 2, 'grid': 3, 'status': 'Finished'},
        ],
        'pit_stops': [
            {'driver_id': 'max_verstappen', 'lap': 15, 'stop': 1, 'duration': 2.5},
            {'driver_id': 'max_verstappen', 'lap': 35, 'stop': 2, 'duration': 2.4},
        ],
        'lap_times': [
            {'driver_id': 'max_verstappen', 'lap': i, 'position': 1, 'time_millis': 95000 + i * 50}
            for i in range(1, 58)
        ]
    }
    
    # Generate features
    sequence, metadata = engineer.prepare_race_sequence(sample_race, 'max_verstappen')
    pit_features = engineer.prepare_pit_stop_features(sample_race, 30, 'max_verstappen')
    strategy_features = engineer.create_strategy_embedding_features(sample_race, 'max_verstappen')
    
    return {
        'sequence': sequence,
        'pit_features': pit_features,
        'strategy_features': strategy_features,
        'metadata': metadata
    }


if __name__ == "__main__":
    # Test feature engineering
    sample = create_sample_features()
    
    print("Feature Engineering Test Results:")
    print(f"  Sequence shape: {sample['sequence'].shape}")
    print(f"  Pit features shape: {sample['pit_features'].shape}")
    print(f"  Strategy description: {sample['strategy_features']['description'][:100]}...")
    print(f"  Metadata: {sample['metadata']}")
