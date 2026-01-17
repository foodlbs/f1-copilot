"""
F1 Data Collection Module

Collects Formula 1 data from multiple sources:
- Ergast API: Historical race data (1950-2024)
- FastF1: Detailed telemetry and timing data
- OpenF1 API: Real-time race data

Author: F1 Race Strategy Analyzer
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import numpy as np

try:
    import fastf1
    fastf1.Cache.enable_cache('./cache/fastf1')
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False
    logging.warning("FastF1 not available. Some features will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RaceResult:
    """Data class for race results"""
    position: int
    driver_id: str
    driver_name: str
    constructor: str
    grid: int
    laps: int
    status: str
    time_millis: Optional[int]
    fastest_lap: Optional[int]
    fastest_lap_time: Optional[str]
    points: float


@dataclass
class PitStop:
    """Data class for pit stop information"""
    driver_id: str
    lap: int
    stop_number: int
    duration: float
    time_of_day: str


@dataclass
class LapTime:
    """Data class for lap timing data"""
    driver_id: str
    lap: int
    position: int
    time_millis: int
    sector_1: Optional[float]
    sector_2: Optional[float]
    sector_3: Optional[float]


@dataclass
class RaceData:
    """Complete race data container"""
    season: int
    round: int
    race_name: str
    circuit_id: str
    circuit_name: str
    date: str
    results: List[RaceResult]
    pit_stops: List[PitStop]
    lap_times: List[LapTime]
    weather: Optional[Dict[str, Any]]
    telemetry: Optional[Dict[str, Any]]


class ErgastAPI:
    """Client for the Ergast F1 API"""
    
    BASE_URL = "https://ergast.com/api/f1"
    
    def __init__(self, rate_limit: float = 0.5):
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'F1-Race-Strategy-Analyzer/1.0'
        })
    
    def _rate_limit_wait(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make a rate-limited request to the API"""
        self._rate_limit_wait()
        
        url = f"{self.BASE_URL}/{endpoint}.json"
        params = params or {}
        params['limit'] = params.get('limit', 1000)
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def get_season_schedule(self, year: int) -> List[Dict]:
        """Get the race schedule for a season"""
        data = self._make_request(f"{year}")
        races = data.get('MRData', {}).get('RaceTable', {}).get('Races', [])
        return races
    
    def get_race_results(self, year: int, round_num: int) -> List[Dict]:
        """Get results for a specific race"""
        data = self._make_request(f"{year}/{round_num}/results")
        races = data.get('MRData', {}).get('RaceTable', {}).get('Races', [])
        if races:
            return races[0].get('Results', [])
        return []
    
    def get_qualifying_results(self, year: int, round_num: int) -> List[Dict]:
        """Get qualifying results for a specific race"""
        data = self._make_request(f"{year}/{round_num}/qualifying")
        races = data.get('MRData', {}).get('RaceTable', {}).get('Races', [])
        if races:
            return races[0].get('QualifyingResults', [])
        return []
    
    def get_pit_stops(self, year: int, round_num: int) -> List[Dict]:
        """Get pit stop data for a race"""
        data = self._make_request(f"{year}/{round_num}/pitstops")
        races = data.get('MRData', {}).get('RaceTable', {}).get('Races', [])
        if races:
            return races[0].get('PitStops', [])
        return []
    
    def get_lap_times(self, year: int, round_num: int, lap: Optional[int] = None) -> List[Dict]:
        """Get lap times for a race"""
        endpoint = f"{year}/{round_num}/laps"
        if lap:
            endpoint += f"/{lap}"
        
        data = self._make_request(endpoint)
        races = data.get('MRData', {}).get('RaceTable', {}).get('Races', [])
        if races:
            return races[0].get('Laps', [])
        return []
    
    def get_driver_standings(self, year: int, round_num: Optional[int] = None) -> List[Dict]:
        """Get driver standings"""
        endpoint = f"{year}"
        if round_num:
            endpoint += f"/{round_num}"
        endpoint += "/driverStandings"
        
        data = self._make_request(endpoint)
        standings_lists = data.get('MRData', {}).get('StandingsTable', {}).get('StandingsLists', [])
        if standings_lists:
            return standings_lists[0].get('DriverStandings', [])
        return []
    
    def get_constructor_standings(self, year: int, round_num: Optional[int] = None) -> List[Dict]:
        """Get constructor standings"""
        endpoint = f"{year}"
        if round_num:
            endpoint += f"/{round_num}"
        endpoint += "/constructorStandings"
        
        data = self._make_request(endpoint)
        standings_lists = data.get('MRData', {}).get('StandingsTable', {}).get('StandingsLists', [])
        if standings_lists:
            return standings_lists[0].get('ConstructorStandings', [])
        return []
    
    def get_circuits(self, year: Optional[int] = None) -> List[Dict]:
        """Get circuit information"""
        endpoint = "circuits" if not year else f"{year}/circuits"
        data = self._make_request(endpoint)
        return data.get('MRData', {}).get('CircuitTable', {}).get('Circuits', [])
    
    def get_drivers(self, year: Optional[int] = None) -> List[Dict]:
        """Get driver information"""
        endpoint = "drivers" if not year else f"{year}/drivers"
        data = self._make_request(endpoint)
        return data.get('MRData', {}).get('DriverTable', {}).get('Drivers', [])
    
    def get_constructors(self, year: Optional[int] = None) -> List[Dict]:
        """Get constructor information"""
        endpoint = "constructors" if not year else f"{year}/constructors"
        data = self._make_request(endpoint)
        return data.get('MRData', {}).get('ConstructorTable', {}).get('Constructors', [])


class OpenF1API:
    """Client for the OpenF1 API (real-time data)"""
    
    BASE_URL = "https://api.openf1.org/v1"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'F1-Race-Strategy-Analyzer/1.0'
        })
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> List[Dict]:
        """Make a request to the OpenF1 API"""
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenF1 API request failed: {e}")
            return []
    
    def get_sessions(self, year: Optional[int] = None, 
                     session_type: Optional[str] = None) -> List[Dict]:
        """Get session information"""
        params = {}
        if year:
            params['year'] = year
        if session_type:
            params['session_type'] = session_type
        return self._make_request("sessions", params)
    
    def get_drivers(self, session_key: Optional[int] = None) -> List[Dict]:
        """Get driver information for a session"""
        params = {}
        if session_key:
            params['session_key'] = session_key
        return self._make_request("drivers", params)
    
    def get_car_data(self, session_key: int, driver_number: Optional[int] = None,
                     speed_gt: Optional[int] = None) -> List[Dict]:
        """Get car telemetry data"""
        params = {'session_key': session_key}
        if driver_number:
            params['driver_number'] = driver_number
        if speed_gt:
            params['speed>'] = speed_gt
        return self._make_request("car_data", params)
    
    def get_intervals(self, session_key: int, 
                      driver_number: Optional[int] = None) -> List[Dict]:
        """Get interval data between cars"""
        params = {'session_key': session_key}
        if driver_number:
            params['driver_number'] = driver_number
        return self._make_request("intervals", params)
    
    def get_laps(self, session_key: int, 
                 driver_number: Optional[int] = None) -> List[Dict]:
        """Get lap data"""
        params = {'session_key': session_key}
        if driver_number:
            params['driver_number'] = driver_number
        return self._make_request("laps", params)
    
    def get_pit(self, session_key: int,
                driver_number: Optional[int] = None) -> List[Dict]:
        """Get pit stop data"""
        params = {'session_key': session_key}
        if driver_number:
            params['driver_number'] = driver_number
        return self._make_request("pit", params)
    
    def get_position(self, session_key: int,
                     driver_number: Optional[int] = None) -> List[Dict]:
        """Get position data"""
        params = {'session_key': session_key}
        if driver_number:
            params['driver_number'] = driver_number
        return self._make_request("position", params)
    
    def get_stints(self, session_key: int,
                   driver_number: Optional[int] = None) -> List[Dict]:
        """Get stint data (tire compounds)"""
        params = {'session_key': session_key}
        if driver_number:
            params['driver_number'] = driver_number
        return self._make_request("stints", params)
    
    def get_weather(self, session_key: int) -> List[Dict]:
        """Get weather data for a session"""
        params = {'session_key': session_key}
        return self._make_request("weather", params)


class FastF1Collector:
    """Collector for FastF1 telemetry data"""
    
    def __init__(self, cache_dir: str = './cache/fastf1'):
        if not FASTF1_AVAILABLE:
            raise ImportError("FastF1 is not installed. Install with: pip install fastf1")
        
        os.makedirs(cache_dir, exist_ok=True)
        fastf1.Cache.enable_cache(cache_dir)
    
    def get_session(self, year: int, race: str | int, 
                    session_type: str = 'R') -> Any:
        """Load a FastF1 session"""
        try:
            session = fastf1.get_session(year, race, session_type)
            session.load()
            return session
        except Exception as e:
            logger.error(f"Failed to load FastF1 session: {e}")
            return None
    
    def get_telemetry(self, session: Any, driver: str) -> Optional[pd.DataFrame]:
        """Get telemetry data for a driver"""
        try:
            driver_laps = session.laps.pick_driver(driver)
            telemetry = driver_laps.get_telemetry()
            return telemetry
        except Exception as e:
            logger.error(f"Failed to get telemetry for {driver}: {e}")
            return None
    
    def get_lap_data(self, session: Any) -> pd.DataFrame:
        """Get lap data for all drivers"""
        return session.laps
    
    def get_weather_data(self, session: Any) -> pd.DataFrame:
        """Get weather data for the session"""
        return session.weather_data
    
    def get_race_control_messages(self, session: Any) -> pd.DataFrame:
        """Get race control messages (flags, penalties, etc.)"""
        return session.race_control_messages
    
    def get_driver_info(self, session: Any) -> Dict:
        """Get driver information for the session"""
        return session.results.to_dict('records')
    
    def analyze_tire_degradation(self, session: Any, driver: str) -> Dict[str, Any]:
        """Analyze tire degradation for a driver"""
        try:
            driver_laps = session.laps.pick_driver(driver)
            
            # Group by stint (tire compound)
            stints = driver_laps.groupby('Stint')
            
            degradation_data = {}
            for stint_num, stint_laps in stints:
                lap_times = stint_laps['LapTime'].dt.total_seconds().dropna()
                if len(lap_times) > 1:
                    # Calculate degradation as slope of lap times
                    x = np.arange(len(lap_times))
                    slope, _ = np.polyfit(x, lap_times, 1)
                    
                    compound = stint_laps['Compound'].iloc[0] if 'Compound' in stint_laps else 'Unknown'
                    degradation_data[f"stint_{stint_num}"] = {
                        'compound': compound,
                        'laps': len(lap_times),
                        'degradation_per_lap': slope,
                        'avg_lap_time': lap_times.mean(),
                        'best_lap_time': lap_times.min()
                    }
            
            return degradation_data
        except Exception as e:
            logger.error(f"Failed to analyze tire degradation: {e}")
            return {}


class F1DataCollector:
    """Main data collector combining all data sources"""
    
    def __init__(self, use_fastf1: bool = True):
        self.ergast = ErgastAPI()
        self.openf1 = OpenF1API()
        self.fastf1 = FastF1Collector() if use_fastf1 and FASTF1_AVAILABLE else None
        
        # Create data directories
        os.makedirs('./data/raw', exist_ok=True)
        os.makedirs('./data/processed', exist_ok=True)
        os.makedirs('./cache', exist_ok=True)
    
    def collect_race_data(self, year: int, round_num: int, 
                          include_telemetry: bool = False) -> RaceData:
        """Collect comprehensive race data"""
        logger.info(f"Collecting data for {year} Round {round_num}")
        
        # Get race info
        schedule = self.ergast.get_season_schedule(year)
        race_info = next((r for r in schedule if int(r['round']) == round_num), None)
        
        if not race_info:
            raise ValueError(f"Race not found: {year} Round {round_num}")
        
        # Get race results
        results_raw = self.ergast.get_race_results(year, round_num)
        results = [
            RaceResult(
                position=int(r['position']),
                driver_id=r['Driver']['driverId'],
                driver_name=f"{r['Driver']['givenName']} {r['Driver']['familyName']}",
                constructor=r['Constructor']['name'],
                grid=int(r['grid']),
                laps=int(r['laps']),
                status=r['status'],
                time_millis=int(r['Time']['millis']) if 'Time' in r else None,
                fastest_lap=int(r['FastestLap']['rank']) if 'FastestLap' in r else None,
                fastest_lap_time=r['FastestLap']['Time']['time'] if 'FastestLap' in r else None,
                points=float(r['points'])
            )
            for r in results_raw
        ]
        
        # Get pit stops
        pit_stops_raw = self.ergast.get_pit_stops(year, round_num)
        pit_stops = [
            PitStop(
                driver_id=p['driverId'],
                lap=int(p['lap']),
                stop_number=int(p['stop']),
                duration=float(p['duration']),
                time_of_day=p['time']
            )
            for p in pit_stops_raw
        ]
        
        # Get lap times (sample first 10 laps to avoid too many requests)
        lap_times = []
        laps_data = self.ergast.get_lap_times(year, round_num)
        for lap in laps_data[:20]:  # Limit to first 20 laps
            lap_num = int(lap['number'])
            for timing in lap.get('Timings', []):
                # Convert time string to milliseconds
                time_str = timing['time']
                try:
                    parts = time_str.split(':')
                    if len(parts) == 2:
                        mins, secs = parts
                        time_ms = int(float(mins) * 60000 + float(secs) * 1000)
                    else:
                        time_ms = int(float(time_str) * 1000)
                except:
                    time_ms = 0
                
                lap_times.append(LapTime(
                    driver_id=timing['driverId'],
                    lap=lap_num,
                    position=int(timing['position']),
                    time_millis=time_ms,
                    sector_1=None,
                    sector_2=None,
                    sector_3=None
                ))
        
        # Get weather and telemetry from FastF1 if available
        weather = None
        telemetry = None
        
        if include_telemetry and self.fastf1:
            try:
                session = self.fastf1.get_session(year, round_num, 'R')
                if session:
                    weather_df = self.fastf1.get_weather_data(session)
                    if weather_df is not None and not weather_df.empty:
                        weather = {
                            'track_temp_avg': weather_df['TrackTemp'].mean(),
                            'air_temp_avg': weather_df['AirTemp'].mean(),
                            'humidity_avg': weather_df['Humidity'].mean(),
                            'rainfall': weather_df['Rainfall'].any() if 'Rainfall' in weather_df else False
                        }
            except Exception as e:
                logger.warning(f"Could not get FastF1 data: {e}")
        
        return RaceData(
            season=year,
            round=round_num,
            race_name=race_info['raceName'],
            circuit_id=race_info['Circuit']['circuitId'],
            circuit_name=race_info['Circuit']['circuitName'],
            date=race_info['date'],
            results=results,
            pit_stops=pit_stops,
            lap_times=lap_times,
            weather=weather,
            telemetry=telemetry
        )
    
    def get_seasons_data(self, start_year: int, end_year: int,
                         include_telemetry: bool = False) -> List[Dict]:
        """Collect data for multiple seasons"""
        all_seasons = []
        
        for year in range(start_year, end_year + 1):
            logger.info(f"Collecting season {year}")
            
            schedule = self.ergast.get_season_schedule(year)
            season_data = {
                'season': year,
                'races': []
            }
            
            for race in schedule:
                round_num = int(race['round'])
                try:
                    race_data = self.collect_race_data(year, round_num, include_telemetry)
                    season_data['races'].append(asdict(race_data))
                except Exception as e:
                    logger.error(f"Failed to collect {year} Round {round_num}: {e}")
                    continue
            
            all_seasons.append(season_data)
            logger.info(f"Collected {len(season_data['races'])} races for {year}")
        
        return all_seasons
    
    def save_data(self, data: Any, filename: str, format: str = 'json'):
        """Save collected data to file"""
        filepath = f"./data/raw/{filename}"
        
        if format == 'json':
            with open(f"{filepath}.json", 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format == 'csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(f"{filepath}.csv", index=False)
            else:
                pd.DataFrame(data).to_csv(f"{filepath}.csv", index=False)
        
        logger.info(f"Saved data to {filepath}.{format}")
    
    def load_data(self, filename: str, format: str = 'json') -> Any:
        """Load data from file"""
        filepath = f"./data/raw/{filename}.{format}"
        
        if format == 'json':
            with open(filepath, 'r') as f:
                return json.load(f)
        elif format == 'csv':
            return pd.read_csv(filepath)
    
    def get_historical_strategies(self, circuit_id: str, 
                                   years: int = 5) -> List[Dict]:
        """Get historical race strategies for a circuit"""
        current_year = datetime.now().year
        strategies = []
        
        for year in range(current_year - years, current_year):
            try:
                schedule = self.ergast.get_season_schedule(year)
                race = next((r for r in schedule 
                            if r['Circuit']['circuitId'] == circuit_id), None)
                
                if race:
                    round_num = int(race['round'])
                    pit_stops = self.ergast.get_pit_stops(year, round_num)
                    results = self.ergast.get_race_results(year, round_num)
                    
                    # Analyze pit stop strategies
                    driver_strategies = {}
                    for stop in pit_stops:
                        driver = stop['driverId']
                        if driver not in driver_strategies:
                            driver_strategies[driver] = []
                        driver_strategies[driver].append({
                            'lap': int(stop['lap']),
                            'duration': float(stop['duration'])
                        })
                    
                    # Match with results
                    for result in results[:10]:  # Top 10
                        driver = result['Driver']['driverId']
                        strategies.append({
                            'year': year,
                            'circuit': circuit_id,
                            'driver': driver,
                            'position': int(result['position']),
                            'grid': int(result['grid']),
                            'pit_stops': driver_strategies.get(driver, []),
                            'num_stops': len(driver_strategies.get(driver, []))
                        })
            except Exception as e:
                logger.warning(f"Could not get strategy for {year}: {e}")
        
        return strategies


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='F1 Data Collection')
    parser.add_argument('--year', type=int, default=2024, help='Season year')
    parser.add_argument('--round', type=int, help='Race round number')
    parser.add_argument('--seasons', nargs=2, type=int, help='Start and end year for season collection')
    
    args = parser.parse_args()
    
    collector = F1DataCollector(use_fastf1=False)
    
    if args.seasons:
        data = collector.get_seasons_data(args.seasons[0], args.seasons[1])
        collector.save_data(data, f"seasons_{args.seasons[0]}_{args.seasons[1]}")
    elif args.round:
        data = collector.collect_race_data(args.year, args.round)
        collector.save_data(asdict(data), f"race_{args.year}_{args.round}")
    else:
        # Default: get current season schedule
        schedule = collector.ergast.get_season_schedule(args.year)
        print(f"\n{args.year} F1 Season Schedule:")
        for race in schedule:
            print(f"  Round {race['round']}: {race['raceName']} - {race['date']}")
