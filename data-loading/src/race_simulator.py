"""
F1 Race Simulator

Simulates race progression lap-by-lap with:
- Strategy execution
- Tire degradation
- Pit stop timing
- Position changes
- Real-time strategy updates as sessions progress

Integrates with strategy predictor and vector database.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd

try:
    from .strategy_predictor import F1StrategyPredictor, SessionData, StrategyRecommendation
except ImportError:
    from strategy_predictor import F1StrategyPredictor, SessionData, StrategyRecommendation

logger = logging.getLogger(__name__)


@dataclass
class Driver:
    """Driver in the race"""
    number: str
    name: str
    team: str
    position: int = 1
    grid_position: int = 1

    # Tire data
    current_compound: str = "MEDIUM"
    tire_age: int = 0
    tire_degradation_rate: float = 0.05  # seconds per lap

    # Lap times
    base_lap_time: float = 90.0  # seconds
    current_lap_time: float = 90.0

    # Strategy
    strategy: Optional[StrategyRecommendation] = None
    completed_stops: int = 0
    total_race_time: float = 0.0

    # Flags
    in_pit: bool = False
    dnf: bool = False
    dnf_reason: Optional[str] = None


@dataclass
class RaceState:
    """Current race state"""
    circuit: str
    total_laps: int
    current_lap: int = 0

    # Conditions
    air_temp: float = 25.0
    track_temp: float = 35.0
    weather: str = "Dry"

    # Race control
    safety_car: bool = False
    red_flag: bool = False
    vsc: bool = False  # Virtual Safety Car

    # Track info
    pit_loss_time: float = 20.0  # Time lost in pit stop
    available_compounds: List[str] = field(default_factory=lambda: ["SOFT", "MEDIUM", "HARD"])


class RaceSimulator:
    """Simulates F1 race progression"""

    def __init__(
        self,
        strategy_predictor: Optional[F1StrategyPredictor] = None,
        enable_randomness: bool = True
    ):
        """
        Initialize race simulator

        Args:
            strategy_predictor: Strategy predictor instance
            enable_randomness: Add random variation to simulation
        """
        self.predictor = strategy_predictor or F1StrategyPredictor()
        self.enable_randomness = enable_randomness

        self.drivers: List[Driver] = []
        self.race_state: Optional[RaceState] = None

        self.lap_history: List[Dict[str, Any]] = []
        self.event_log: List[str] = []

        logger.info("Race Simulator initialized")

    def setup_race(
        self,
        circuit: str,
        total_laps: int,
        drivers: List[Dict[str, Any]],
        weather: str = "Dry",
        air_temp: float = 25.0,
        track_temp: float = 35.0
    ):
        """
        Set up race simulation

        Args:
            circuit: Circuit name
            total_laps: Total race laps
            drivers: List of driver configurations
            weather: Weather condition
            air_temp: Air temperature
            track_temp: Track temperature
        """
        self.race_state = RaceState(
            circuit=circuit,
            total_laps=total_laps,
            current_lap=0,
            air_temp=air_temp,
            track_temp=track_temp,
            weather=weather
        )

        # Initialize drivers
        self.drivers = []
        for i, driver_config in enumerate(drivers):
            driver = Driver(
                number=driver_config['number'],
                name=driver_config['name'],
                team=driver_config['team'],
                position=i + 1,
                grid_position=i + 1,
                base_lap_time=driver_config.get('base_lap_time', 90.0 + i * 0.5),
                current_compound=driver_config.get('starting_compound', 'MEDIUM')
            )

            # Generate strategy for each driver
            session_data = self._create_session_data(driver.position)
            driver.strategy = self.predictor.predict_optimal_strategy(session_data, driver.position)

            self.drivers.append(driver)

        self._log_event(f"Race setup complete: {circuit}, {total_laps} laps, {len(self.drivers)} drivers")
        logger.info(f"Race configured: {circuit} ({total_laps} laps)")

    def simulate_race(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Simulate complete race

        Args:
            verbose: Print lap-by-lap updates

        Returns:
            Race results and statistics
        """
        if not self.race_state or not self.drivers:
            raise ValueError("Race not set up. Call setup_race() first.")

        self._log_event("RACE START")
        logger.info(f"Simulating {self.race_state.circuit} race...")

        # Race simulation loop
        for lap in range(1, self.race_state.total_laps + 1):
            self.race_state.current_lap = lap

            # Simulate lap
            self.simulate_lap(verbose=verbose)

            # Check for random events
            if self.enable_randomness:
                self._check_random_events(lap)

        self._log_event("RACE FINISH")

        # Generate results
        results = self._generate_results()

        logger.info("Race simulation complete")
        return results

    def simulate_lap(self, verbose: bool = False):
        """Simulate a single lap"""
        lap = self.race_state.current_lap

        lap_data = {
            'lap': lap,
            'positions': [],
            'lap_times': [],
            'pit_stops': []
        }

        # Update each driver
        for driver in self.drivers:
            if driver.dnf:
                continue

            # Check pit stop
            should_pit = self._should_pit_this_lap(driver, lap)

            if should_pit:
                self._execute_pit_stop(driver, lap)
                lap_data['pit_stops'].append({
                    'driver': driver.name,
                    'lap': lap,
                    'compound': driver.current_compound
                })

            # Calculate lap time
            lap_time = self._calculate_lap_time(driver)
            driver.current_lap_time = lap_time
            driver.total_race_time += lap_time

            # Update tire age
            driver.tire_age += 1

            lap_data['positions'].append({
                'driver': driver.name,
                'position': driver.position,
                'tire_age': driver.tire_age,
                'compound': driver.current_compound
            })

            lap_data['lap_times'].append({
                'driver': driver.name,
                'time': lap_time
            })

        # Update positions based on race time
        self._update_positions()

        # Store lap data
        self.lap_history.append(lap_data)

        if verbose and lap % 5 == 0:
            self._print_lap_update(lap)

    def _should_pit_this_lap(self, driver: Driver, lap: int) -> bool:
        """Determine if driver should pit this lap"""
        if not driver.strategy:
            return False

        # Check if on planned pit lap
        for stint in driver.strategy.stints:
            if stint.get('pit_lap') == lap:
                return True

        # Dynamic pit decision based on tire age
        if driver.tire_age > 25 and driver.current_compound == "SOFT":
            return True

        if driver.tire_age > 35 and driver.current_compound == "MEDIUM":
            return True

        return False

    def _execute_pit_stop(self, driver: Driver, lap: int):
        """Execute pit stop for driver"""
        # Get next compound from strategy
        next_compound = self._get_next_compound(driver, lap)

        # Pit stop time loss
        pit_time = self.race_state.pit_loss_time

        # Add randomness
        if self.enable_randomness:
            pit_time += random.gauss(0, 0.5)  # ±0.5s variation

        driver.total_race_time += pit_time
        driver.current_compound = next_compound
        driver.tire_age = 0
        driver.completed_stops += 1

        self._log_event(
            f"LAP {lap}: {driver.name} pits - {next_compound} tires ({pit_time:.2f}s)"
        )

    def _get_next_compound(self, driver: Driver, lap: int) -> str:
        """Get next tire compound for driver"""
        if not driver.strategy:
            return "MEDIUM"

        # Find current stint
        current_stint_idx = driver.completed_stops

        if current_stint_idx + 1 < len(driver.strategy.stints):
            next_stint = driver.strategy.stints[current_stint_idx + 1]
            return next_stint['compound']

        # Fallback
        return "HARD"

    def _calculate_lap_time(self, driver: Driver) -> float:
        """Calculate lap time for driver"""
        base_time = driver.base_lap_time

        # Tire degradation
        tire_deg = driver.tire_age * driver.tire_degradation_rate
        degradation_multiplier = self._get_compound_degradation_rate(driver.current_compound)
        tire_effect = tire_deg * degradation_multiplier

        # Compound delta (relative to medium)
        compound_delta = self._get_compound_delta(driver.current_compound)

        # Traffic effect (simplified)
        traffic_effect = 0.0
        if driver.position > 10:
            traffic_effect = random.uniform(0, 0.5) if self.enable_randomness else 0.2

        # Safety car
        if self.race_state.safety_car:
            return base_time * 1.5  # Much slower under SC

        # Calculate total lap time
        lap_time = base_time + compound_delta + tire_effect + traffic_effect

        # Add random variation
        if self.enable_randomness:
            lap_time += random.gauss(0, 0.3)

        return max(lap_time, base_time - 5.0)  # Cap improvement

    def _get_compound_delta(self, compound: str) -> float:
        """Get lap time delta for tire compound"""
        deltas = {
            'SOFT': -1.0,    # 1 second faster
            'MEDIUM': 0.0,   # Baseline
            'HARD': 0.5      # 0.5 seconds slower
        }
        return deltas.get(compound.upper(), 0.0)

    def _get_compound_degradation_rate(self, compound: str) -> float:
        """Get degradation rate multiplier for compound"""
        rates = {
            'SOFT': 1.5,   # Degrades faster
            'MEDIUM': 1.0,
            'HARD': 0.7    # Degrades slower
        }
        return rates.get(compound.upper(), 1.0)

    def _update_positions(self):
        """Update driver positions based on race time"""
        # Sort by total race time
        active_drivers = [d for d in self.drivers if not d.dnf]
        active_drivers.sort(key=lambda d: d.total_race_time)

        # Update positions
        for i, driver in enumerate(active_drivers):
            driver.position = i + 1

    def _check_random_events(self, lap: int):
        """Check for random race events"""
        # Small chance of safety car
        if random.random() < 0.02:  # 2% chance per lap
            if not self.race_state.safety_car:
                self._trigger_safety_car(lap)

        # Small chance of DNF
        for driver in self.drivers:
            if not driver.dnf and random.random() < 0.001:  # 0.1% per driver per lap
                self._trigger_dnf(driver, lap)

    def _trigger_safety_car(self, lap: int):
        """Trigger safety car period"""
        self.race_state.safety_car = True
        self._log_event(f"LAP {lap}: SAFETY CAR DEPLOYED")

        # Safety car for 3-5 laps
        sc_duration = random.randint(3, 5)

        # Schedule end of safety car (simplified)
        logger.info(f"Safety car deployed on lap {lap}")

    def _trigger_dnf(self, driver: Driver, lap: int):
        """Trigger DNF for driver"""
        reasons = ["Mechanical", "Accident", "Collision", "Engine"]
        reason = random.choice(reasons)

        driver.dnf = True
        driver.dnf_reason = reason

        self._log_event(f"LAP {lap}: {driver.name} DNF ({reason})")
        logger.info(f"{driver.name} retired from race: {reason}")

    def _generate_results(self) -> Dict[str, Any]:
        """Generate final race results"""
        # Sort drivers by position
        classified = [d for d in self.drivers if not d.dnf]
        dnf = [d for d in self.drivers if d.dnf]

        classified.sort(key=lambda d: d.position)

        results = {
            'circuit': self.race_state.circuit,
            'total_laps': self.race_state.total_laps,
            'weather': self.race_state.weather,
            'classifications': [],
            'dnf': [],
            'statistics': {}
        }

        # Classifications
        for driver in classified:
            results['classifications'].append({
                'position': driver.position,
                'driver': driver.name,
                'team': driver.team,
                'grid': driver.grid_position,
                'stops': driver.completed_stops,
                'total_time': driver.total_race_time,
                'gap_to_leader': driver.total_race_time - classified[0].total_race_time if driver.position > 1 else 0.0
            })

        # DNFs
        for driver in dnf:
            results['dnf'].append({
                'driver': driver.name,
                'reason': driver.dnf_reason
            })

        # Statistics
        results['statistics'] = {
            'total_pit_stops': sum(d.completed_stops for d in self.drivers),
            'dnf_count': len(dnf),
            'safety_cars': 1 if any('SAFETY CAR' in log for log in self.event_log) else 0,
            'average_stops': sum(d.completed_stops for d in classified) / len(classified) if classified else 0
        }

        return results

    def _create_session_data(self, position: int) -> SessionData:
        """Create SessionData for strategy predictor"""
        return SessionData(
            circuit=self.race_state.circuit,
            session_type="Race",
            lap_number=self.race_state.current_lap,
            total_laps=self.race_state.total_laps,
            air_temp=self.race_state.air_temp,
            track_temp=self.race_state.track_temp,
            weather=self.race_state.weather,
            available_compounds=self.race_state.available_compounds,
            position=position
        )

    def get_live_standings(self) -> List[Dict[str, Any]]:
        """Get current race standings"""
        standings = []

        for driver in sorted(self.drivers, key=lambda d: d.position):
            standings.append({
                'position': driver.position,
                'driver': driver.name,
                'team': driver.team,
                'compound': driver.current_compound,
                'tire_age': driver.tire_age,
                'stops': driver.completed_stops,
                'status': 'DNF' if driver.dnf else 'Racing'
            })

        return standings

    def update_strategy_mid_race(self, driver_name: str) -> StrategyRecommendation:
        """
        Update strategy for driver mid-race based on current conditions

        Args:
            driver_name: Driver name

        Returns:
            Updated strategy recommendation
        """
        driver = next((d for d in self.drivers if d.name == driver_name), None)

        if not driver:
            raise ValueError(f"Driver {driver_name} not found")

        # Create current session data
        session_data = self._create_session_data(driver.position)
        session_data.current_compound = driver.current_compound
        session_data.tire_age = driver.tire_age

        # Get updated strategy
        updated_strategy = self.predictor.predict_optimal_strategy(
            session_data,
            driver.position
        )

        driver.strategy = updated_strategy

        self._log_event(
            f"LAP {self.race_state.current_lap}: {driver_name} strategy updated to {updated_strategy.strategy_type}"
        )

        return updated_strategy

    def _log_event(self, message: str):
        """Log race event"""
        self.event_log.append(message)

    def _print_lap_update(self, lap: int):
        """Print lap update"""
        print(f"\n=== LAP {lap}/{self.race_state.total_laps} ===")
        print("Top 5:")
        for i, driver in enumerate(sorted(self.drivers, key=lambda d: d.position)[:5]):
            print(f"  P{driver.position}: {driver.name} - {driver.current_compound} ({driver.tire_age} laps)")

    def export_lap_chart(self) -> pd.DataFrame:
        """Export lap-by-lap position chart"""
        data = []

        for lap_data in self.lap_history:
            for pos_data in lap_data['positions']:
                data.append({
                    'lap': lap_data['lap'],
                    'driver': pos_data['driver'],
                    'position': pos_data['position'],
                    'tire_age': pos_data['tire_age'],
                    'compound': pos_data['compound']
                })

        return pd.DataFrame(data)

    def get_event_log(self) -> List[str]:
        """Get complete event log"""
        return self.event_log


if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    load_dotenv()

    # Initialize simulator
    simulator = RaceSimulator(enable_randomness=True)

    # Example driver grid
    drivers = [
        {'number': '1', 'name': 'Max Verstappen', 'team': 'Red Bull', 'base_lap_time': 78.0},
        {'number': '44', 'name': 'Lewis Hamilton', 'team': 'Mercedes', 'base_lap_time': 78.2},
        {'number': '16', 'name': 'Charles Leclerc', 'team': 'Ferrari', 'base_lap_time': 78.3},
        {'number': '11', 'name': 'Sergio Perez', 'team': 'Red Bull', 'base_lap_time': 78.5},
        {'number': '63', 'name': 'George Russell', 'team': 'Mercedes', 'base_lap_time': 78.6},
    ]

    # Setup race
    simulator.setup_race(
        circuit="Monza",
        total_laps=53,
        drivers=drivers,
        weather="Dry",
        track_temp=35.0
    )

    # Simulate race
    results = simulator.simulate_race(verbose=True)

    # Print results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    for classification in results['classifications']:
        gap = f"+{classification['gap_to_leader']:.3f}s" if classification['position'] > 1 else "Winner"
        print(f"P{classification['position']}: {classification['driver']} ({classification['team']}) - {gap}")

    print(f"\nTotal pit stops: {results['statistics']['total_pit_stops']}")
    print(f"DNFs: {results['statistics']['dnf_count']}")
