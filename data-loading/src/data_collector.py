"""
F1 Data Collection Module

Collects comprehensive Formula 1 data from 2017 onwards including:
- Race results and classifications
- Lap times and sector times
- Pit stop strategies
- Tire compound usage
- Weather conditions
- Track information
- Telemetry data (speed, throttle, brake, DRS)
- Driver and team standings

Uses FastF1 for detailed telemetry and Ergast API as fallback.
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import time

import fastf1
import pandas as pd
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RaceWeekendData:
    """Complete race weekend data structure"""
    season: int
    round: int
    race_name: str
    circuit_name: str
    country: str
    date: str

    # Results
    race_results: List[Dict[str, Any]]
    qualifying_results: Optional[List[Dict[str, Any]]] = None
    sprint_results: Optional[List[Dict[str, Any]]] = None

    # Strategy & Performance
    pit_stops: List[Dict[str, Any]] = None
    lap_times: List[Dict[str, Any]] = None
    tire_strategies: List[Dict[str, Any]] = None

    # Conditions
    weather_data: Optional[Dict[str, Any]] = None
    track_status: Optional[List[Dict[str, Any]]] = None

    # Telemetry (sampled)
    fastest_laps_telemetry: Optional[List[Dict[str, Any]]] = None

    # Metadata
    total_laps: int = 0
    safety_cars: int = 0
    red_flags: int = 0


class F1DataCollector:
    """Collects and processes F1 race data"""

    def __init__(self, cache_dir: str = "./cache/f1_data", start_year: int = 2017, force_redownload: bool = False):
        """
        Initialize F1 data collector

        Args:
            cache_dir: Directory for caching downloaded data
            start_year: Starting year for data collection (default: 2017)
            force_redownload: If True, ignore cached data and re-download everything
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.start_year = start_year
        self.current_year = datetime.now().year
        self.force_redownload = force_redownload

        # Enable FastF1 cache
        fastf1_cache = Path("./cache/fastf1")
        fastf1_cache.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(fastf1_cache))

        if force_redownload:
            logger.info(f"F1 Data Collector initialized (Years: {start_year}-{self.current_year}) - FORCE REDOWNLOAD MODE")
        else:
            logger.info(f"F1 Data Collector initialized (Years: {start_year}-{self.current_year})")

    def collect_all_seasons(self, years: Optional[List[int]] = None) -> List[RaceWeekendData]:
        """
        Collect data for all seasons from start_year to current

        Args:
            years: Specific years to collect (if None, collects all from start_year)

        Returns:
            List of all race weekend data
        """
        if years is None:
            years = list(range(self.start_year, self.current_year + 1))

        all_race_data = []

        for year in tqdm(years, desc="Collecting seasons"):
            logger.info(f"Collecting data for {year} season...")
            season_data = self.collect_season(year)
            all_race_data.extend(season_data)

            # Save intermediate results (only if we have data)
            if season_data:
                self._save_season_cache(year, season_data)

        logger.info(f"Total races collected: {len(all_race_data)}")
        return all_race_data

    def collect_season(self, year: int) -> List[RaceWeekendData]:
        """
        Collect all race weekends for a specific season

        Args:
            year: Season year

        Returns:
            List of race weekend data for the season
        """
        # Check cache first (unless force redownload)
        if not self.force_redownload:
            cached_data = self._load_season_cache(year)
            if cached_data is not None and len(cached_data) > 0:
                logger.info(f"✓ Loaded {year} season from cache ({len(cached_data)} races)")
                return cached_data
            elif cached_data is not None and len(cached_data) == 0:
                logger.warning(f"Empty cache for {year}, re-downloading...")
        else:
            logger.info(f"Skipping cache for {year} (force redownload enabled)")

        season_data = []

        try:
            # Get season schedule
            schedule = fastf1.get_event_schedule(year)

            for idx, event in tqdm(schedule.iterrows(), total=len(schedule), desc=f"{year} Races"):
                # Skip testing and non-race events
                if event['EventFormat'] == 'testing':
                    continue

                try:
                    # Check if we already have this race cached
                    race_cache_file = self.cache_dir / f"race_{year}_R{event['RoundNumber']}.json"

                    if not self.force_redownload and race_cache_file.exists():
                        # Load from race cache
                        try:
                            with open(race_cache_file, 'r') as f:
                                race_dict = json.load(f)
                                race_data = RaceWeekendData(**race_dict)
                                season_data.append(race_data)
                                logger.debug(f"Loaded {year} R{event['RoundNumber']} from cache")
                                continue
                        except Exception:
                            # If cache load fails, re-download
                            pass

                    # Download race data
                    race_data = self.collect_race_weekend(year, event['RoundNumber'])
                    if race_data:
                        season_data.append(race_data)

                        # Save individual race cache
                        self._save_race_cache(year, event['RoundNumber'], race_data)

                    # Rate limiting
                    time.sleep(1)

                except Exception as e:
                    logger.error(f"Error collecting {year} Round {event['RoundNumber']}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error loading {year} schedule: {e}")

        return season_data

    def collect_race_weekend(self, year: int, round_num: int) -> Optional[RaceWeekendData]:
        """
        Collect comprehensive data for a single race weekend

        Args:
            year: Season year
            round_num: Round number

        Returns:
            Complete race weekend data
        """
        try:
            # Load race session
            race = fastf1.get_session(year, round_num, 'R')

            try:
                race.load()
            except Exception as load_error:
                # Some sessions don't have full data available (especially older races)
                logger.warning(f"Partial data load for {year} R{round_num}: {load_error}")
                # Continue anyway - we can still get basic results

            # Get event info
            event = race.event

            # Collect all session data (with graceful degradation)
            race_results = self._get_race_results(race)
            qualifying_results = self._get_qualifying_results(year, round_num)
            sprint_results = self._get_sprint_results(year, round_num)

            # Strategy data
            pit_stops = self._get_pit_stops(race)
            lap_times = self._get_lap_times(race)
            tire_strategies = self._get_tire_strategies(race)

            # Conditions
            weather_data = self._get_weather_data(race)
            track_status = self._get_track_status(race)

            # Telemetry (sample fastest laps)
            fastest_laps_telemetry = self._get_fastest_laps_telemetry(race)

            # Count incidents
            safety_cars, red_flags = self._count_incidents(race)

            race_weekend = RaceWeekendData(
                season=year,
                round=round_num,
                race_name=event['EventName'],
                circuit_name=event['Location'],
                country=event['Country'],
                date=str(event['EventDate'].date()),
                race_results=race_results,
                qualifying_results=qualifying_results,
                sprint_results=sprint_results,
                pit_stops=pit_stops,
                lap_times=lap_times,
                tire_strategies=tire_strategies,
                weather_data=weather_data,
                track_status=track_status,
                fastest_laps_telemetry=fastest_laps_telemetry,
                total_laps=race.total_laps,
                safety_cars=safety_cars,
                red_flags=red_flags
            )

            # Log collection status
            if race_results:
                logger.info(f"✓ Collected {year} {event['EventName']} ({len(race_results)} drivers)")
            else:
                logger.warning(f"⚠ Collected {year} {event['EventName']} (limited data)")

            return race_weekend

        except Exception as e:
            logger.warning(f"⚠ Skipping {year} Round {round_num}: {str(e)[:100]}")
            return None

    def _get_race_results(self, session) -> List[Dict[str, Any]]:
        """Extract race results"""
        results = []

        for idx, driver in session.results.iterrows():
            # Handle FastestLap field safely (not always present)
            fastest_lap = None
            if 'FastestLap' in driver.index and pd.notna(driver['FastestLap']):
                fastest_lap = str(driver['FastestLap'])

            results.append({
                'position': int(driver['Position']) if pd.notna(driver['Position']) else None,
                'driver_number': str(driver['DriverNumber']),
                'driver_code': str(driver['Abbreviation']),
                'driver_name': str(driver['FullName']),
                'team': str(driver['TeamName']),
                'grid_position': int(driver['GridPosition']) if pd.notna(driver['GridPosition']) else None,
                'status': str(driver['Status']),
                'points': float(driver['Points']) if pd.notna(driver['Points']) else 0.0,
                'time': str(driver['Time']) if pd.notna(driver['Time']) else None,
                'fastest_lap': fastest_lap,
            })

        return results

    def _get_qualifying_results(self, year: int, round_num: int) -> Optional[List[Dict[str, Any]]]:
        """Extract qualifying results"""
        try:
            quali = fastf1.get_session(year, round_num, 'Q')
            quali.load()

            results = []
            for idx, driver in quali.results.iterrows():
                results.append({
                    'position': int(driver['Position']) if pd.notna(driver['Position']) else None,
                    'driver_code': str(driver['Abbreviation']),
                    'driver_name': str(driver['FullName']),
                    'team': str(driver['TeamName']),
                    'q1': str(driver['Q1']) if pd.notna(driver['Q1']) else None,
                    'q2': str(driver['Q2']) if pd.notna(driver['Q2']) else None,
                    'q3': str(driver['Q3']) if pd.notna(driver['Q3']) else None,
                })

            return results

        except Exception as e:
            logger.warning(f"Could not load qualifying: {e}")
            return None

    def _get_sprint_results(self, year: int, round_num: int) -> Optional[List[Dict[str, Any]]]:
        """Extract sprint race results if available"""
        try:
            sprint = fastf1.get_session(year, round_num, 'S')
            sprint.load()

            results = []
            for idx, driver in sprint.results.iterrows():
                results.append({
                    'position': int(driver['Position']) if pd.notna(driver['Position']) else None,
                    'driver_code': str(driver['Abbreviation']),
                    'driver_name': str(driver['FullName']),
                    'team': str(driver['TeamName']),
                    'grid_position': int(driver['GridPosition']) if pd.notna(driver['GridPosition']) else None,
                    'points': float(driver['Points']) if pd.notna(driver['Points']) else 0.0,
                })

            return results

        except Exception:
            # Sprint not available for this race
            return None

    def _get_pit_stops(self, session) -> List[Dict[str, Any]]:
        """Extract pit stop data"""
        pit_stops = []

        try:
            # Check if laps data is available
            if not hasattr(session, 'laps') or session.laps is None or session.laps.empty:
                return []

            laps = session.laps

            for driver in session.drivers:
                driver_laps = laps.pick_drivers(driver)

                for idx, lap in driver_laps.iterrows():
                    if pd.notna(lap['PitInTime']) and pd.notna(lap['PitOutTime']):
                        pit_duration = (lap['PitOutTime'] - lap['PitInTime']).total_seconds()

                        pit_stops.append({
                            'driver': str(lap['Driver']),
                            'lap': int(lap['LapNumber']),
                            'pit_duration': float(pit_duration),
                            'compound_before': str(driver_laps[driver_laps['LapNumber'] == lap['LapNumber'] - 1]['Compound'].values[0]) if lap['LapNumber'] > 1 else None,
                            'compound_after': str(lap['Compound']) if pd.notna(lap['Compound']) else None,
                        })

        except Exception as e:
            logger.warning(f"Error extracting pit stops: {e}")

        return pit_stops

    def _get_lap_times(self, session) -> List[Dict[str, Any]]:
        """Extract lap time data (sampled to reduce size)"""
        lap_times = []

        try:
            # Check if laps data is available
            if not hasattr(session, 'laps') or session.laps is None or session.laps.empty:
                return []

            laps = session.laps

            # Sample every 5th lap to reduce data size
            for driver in session.drivers:
                driver_laps = laps.pick_drivers(driver)

                for idx, lap in driver_laps.iterrows():
                    # Include all laps for first 10, then every 5th
                    if lap['LapNumber'] <= 10 or lap['LapNumber'] % 5 == 0:
                        lap_times.append({
                            'driver': str(lap['Driver']),
                            'lap': int(lap['LapNumber']),
                            'time': float(lap['LapTime'].total_seconds()) if pd.notna(lap['LapTime']) else None,
                            'sector1': float(lap['Sector1Time'].total_seconds()) if pd.notna(lap['Sector1Time']) else None,
                            'sector2': float(lap['Sector2Time'].total_seconds()) if pd.notna(lap['Sector2Time']) else None,
                            'sector3': float(lap['Sector3Time'].total_seconds()) if pd.notna(lap['Sector3Time']) else None,
                            'compound': str(lap['Compound']) if pd.notna(lap['Compound']) else None,
                            'tire_life': int(lap['TyreLife']) if pd.notna(lap['TyreLife']) else None,
                            'track_status': str(lap['TrackStatus']) if pd.notna(lap['TrackStatus']) else None,
                        })

        except Exception as e:
            logger.warning(f"Error extracting lap times: {e}")

        return lap_times

    def _get_tire_strategies(self, session) -> List[Dict[str, Any]]:
        """Extract tire strategy for each driver"""
        strategies = []

        try:
            # Check if laps data is available
            if not hasattr(session, 'laps') or session.laps is None or session.laps.empty:
                return []

            laps = session.laps

            for driver in session.drivers:
                driver_laps = laps.pick_drivers(driver)
                stints = []

                current_compound = None
                stint_start = 1

                for idx, lap in driver_laps.iterrows():
                    compound = str(lap['Compound']) if pd.notna(lap['Compound']) else None

                    if compound != current_compound and compound is not None:
                        if current_compound is not None:
                            stints.append({
                                'compound': current_compound,
                                'start_lap': stint_start,
                                'end_lap': int(lap['LapNumber']) - 1,
                                'stint_length': int(lap['LapNumber']) - stint_start
                            })

                        current_compound = compound
                        stint_start = int(lap['LapNumber'])

                # Add final stint
                if current_compound is not None:
                    stints.append({
                        'compound': current_compound,
                        'start_lap': stint_start,
                        'end_lap': int(driver_laps.iloc[-1]['LapNumber']),
                        'stint_length': int(driver_laps.iloc[-1]['LapNumber']) - stint_start + 1
                    })

                strategies.append({
                    'driver': str(driver),
                    'stints': stints,
                    'total_stops': len(stints) - 1
                })

        except Exception as e:
            logger.warning(f"Error extracting tire strategies: {e}")

        return strategies

    def _get_weather_data(self, session) -> Optional[Dict[str, Any]]:
        """Extract weather data"""
        try:
            # Check if weather data is available
            if not hasattr(session, 'weather_data'):
                return None

            weather = session.weather_data

            if weather is not None and not weather.empty:
                return {
                    'air_temp_avg': float(weather['AirTemp'].mean()),
                    'track_temp_avg': float(weather['TrackTemp'].mean()),
                    'humidity_avg': float(weather['Humidity'].mean()),
                    'pressure_avg': float(weather['Pressure'].mean()),
                    'rainfall': bool(weather['Rainfall'].any()),
                    'wind_speed_avg': float(weather['WindSpeed'].mean()) if 'WindSpeed' in weather.columns else None,
                }

        except Exception as e:
            logger.warning(f"Error extracting weather: {e}")

        return None

    def _get_track_status(self, session) -> Optional[List[Dict[str, Any]]]:
        """Extract track status changes (safety car, yellow flags, etc.)"""
        try:
            # Check if laps data is available
            if not hasattr(session, 'laps') or session.laps is None or session.laps.empty:
                return None

            laps = session.laps

            # Get unique track status changes
            status_changes = []
            prev_status = None

            for lap_num in sorted(laps['LapNumber'].unique()):
                lap_data = laps[laps['LapNumber'] == lap_num]
                status = lap_data.iloc[0]['TrackStatus'] if 'TrackStatus' in lap_data.columns else None

                if status != prev_status and pd.notna(status):
                    status_changes.append({
                        'lap': int(lap_num),
                        'status': str(status)
                    })
                    prev_status = status

            return status_changes if status_changes else None

        except Exception as e:
            logger.warning(f"Error extracting track status: {e}")

        return None

    def _get_fastest_laps_telemetry(self, session) -> Optional[List[Dict[str, Any]]]:
        """Extract telemetry for fastest laps (top 5 drivers)"""
        telemetry_data = []

        try:
            # Check if laps data is available
            if not hasattr(session, 'laps') or session.laps is None or session.laps.empty:
                return None

            laps = session.laps

            # Get top 5 fastest laps
            fastest_laps = laps.sort_values('LapTime').head(5)

            for idx, lap in fastest_laps.iterrows():
                try:
                    telemetry = lap.get_telemetry()

                    # Sample telemetry (every 10th point to reduce size)
                    sampled = telemetry.iloc[::10]

                    telemetry_data.append({
                        'driver': str(lap['Driver']),
                        'lap': int(lap['LapNumber']),
                        'lap_time': float(lap['LapTime'].total_seconds()),
                        'speed_max': float(sampled['Speed'].max()),
                        'speed_avg': float(sampled['Speed'].mean()),
                        'throttle_avg': float(sampled['Throttle'].mean()),
                        'brake_points': int((sampled['Brake'] > 0).sum()),
                        'drs_usage': float((sampled['DRS'] > 0).sum() / len(sampled) * 100) if 'DRS' in sampled.columns else 0,
                    })

                except Exception:
                    continue

        except Exception as e:
            logger.warning(f"Error extracting telemetry: {e}")

        return telemetry_data if telemetry_data else None

    def _count_incidents(self, session) -> tuple:
        """Count safety cars and red flags"""
        safety_cars = 0
        red_flags = 0

        try:
            # Check if laps data is available
            if not hasattr(session, 'laps') or session.laps is None or session.laps.empty:
                return 0, 0

            laps = session.laps

            if 'TrackStatus' in laps.columns:
                # Safety car typically indicated by track status 4 or 6
                safety_cars = int(((laps['TrackStatus'] == '4') | (laps['TrackStatus'] == '6')).sum() > 0)

                # Red flag typically indicated by track status 5
                red_flags = int((laps['TrackStatus'] == '5').sum() > 0)

        except Exception as e:
            logger.warning(f"Error counting incidents: {e}")

        return safety_cars, red_flags

    def _save_season_cache(self, year: int, data: List[RaceWeekendData]):
        """Save season data to cache"""
        cache_file = self.cache_dir / f"season_{year}.json"

        try:
            with open(cache_file, 'w') as f:
                json.dump([asdict(race) for race in data], f, indent=2)
            logger.info(f"Saved {year} season to cache")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def _load_season_cache(self, year: int) -> Optional[List[RaceWeekendData]]:
        """Load season data from cache"""
        cache_file = self.cache_dir / f"season_{year}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            return [RaceWeekendData(**race) for race in data]
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return None

    def _save_race_cache(self, year: int, round_num: int, race_data: RaceWeekendData):
        """Save individual race data to cache"""
        cache_file = self.cache_dir / f"race_{year}_R{round_num}.json"

        try:
            with open(cache_file, 'w') as f:
                json.dump(asdict(race_data), f, indent=2)
            logger.debug(f"Cached {year} R{round_num}")
        except Exception as e:
            logger.error(f"Error saving race cache: {e}")

    def export_to_json(self, data: List[RaceWeekendData], output_file: str):
        """Export collected data to JSON file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump([asdict(race) for race in data], f, indent=2)

        logger.info(f"Exported {len(data)} races to {output_file}")


if __name__ == "__main__":
    # Example usage
    collector = F1DataCollector(start_year=2017)

    # Collect all seasons
    all_data = collector.collect_all_seasons()

    # Export to JSON
    collector.export_to_json(all_data, "data/f1_race_data.json")

    print(f"\nCollected {len(all_data)} race weekends")
    print(f"Date range: {all_data[0].date} to {all_data[-1].date}")
