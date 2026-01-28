"""
OpenF1 API Data Collector

Collects real-time and historical F1 data from OpenF1 API.
Provides additional data not available in FastF1 or CSV:
- Real-time position tracking
- Overtake data
- Detailed pit stop timing
- Championship progression during races
- Team radio transcripts

API Documentation: https://openf1.org/
"""

import logging
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


class OpenF1Collector:
    """Collect data from OpenF1 API"""

    BASE_URL = "https://api.openf1.org/v1"

    def __init__(self, rate_limit_delay: float = 0.2):
        """
        Initialize OpenF1 collector

        Args:
            rate_limit_delay: Delay between API calls (seconds)
        """
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'F1FantasyApp/1.0'
        })

        logger.info("OpenF1 collector initialized")

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Make API request with error handling

        Args:
            endpoint: API endpoint (e.g., '/drivers')
            params: Query parameters

        Returns:
            JSON response as list of dicts
        """
        url = f"{self.BASE_URL}{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            # Rate limiting
            time.sleep(self.rate_limit_delay)

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {url} - {e}")
            return []

    def get_meetings(
        self,
        year: Optional[int] = None,
        country_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get race meetings (events)

        Args:
            year: Filter by year
            country_name: Filter by country

        Returns:
            List of meeting data
        """
        params = {}
        if year:
            params['year'] = year
        if country_name:
            params['country_name'] = country_name

        return self._make_request('/meetings', params)

    def get_sessions(
        self,
        meeting_key: Optional[int] = None,
        session_name: Optional[str] = None,
        year: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get session data (FP1, FP2, FP3, Qualifying, Sprint, Race)

        Args:
            meeting_key: Filter by meeting
            session_name: Filter by session type
            year: Filter by year

        Returns:
            List of session data
        """
        params = {}
        if meeting_key:
            params['meeting_key'] = meeting_key
        if session_name:
            params['session_name'] = session_name
        if year:
            params['year'] = year

        return self._make_request('/sessions', params)

    def get_drivers(
        self,
        session_key: Optional[int] = None,
        driver_number: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get driver information

        Args:
            session_key: Filter by session
            driver_number: Filter by driver number

        Returns:
            List of driver data
        """
        params = {}
        if session_key:
            params['session_key'] = session_key
        if driver_number:
            params['driver_number'] = driver_number

        return self._make_request('/drivers', params)

    def get_session_results(
        self,
        session_key: int,
        driver_number: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get final session results

        Args:
            session_key: Session identifier
            driver_number: Filter by driver

        Returns:
            List of results with positions, gaps, points
        """
        params = {'session_key': session_key}
        if driver_number:
            params['driver_number'] = driver_number

        return self._make_request('/session_result', params)

    def get_laps(
        self,
        session_key: int,
        driver_number: Optional[int] = None,
        lap_number: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get lap data with sector times

        Args:
            session_key: Session identifier
            driver_number: Filter by driver
            lap_number: Filter by lap

        Returns:
            List of lap data with sector times
        """
        params = {'session_key': session_key}
        if driver_number:
            params['driver_number'] = driver_number
        if lap_number:
            params['lap_number'] = lap_number

        return self._make_request('/laps', params)

    def get_pit_stops(
        self,
        session_key: int,
        driver_number: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get pit stop data with durations

        Args:
            session_key: Session identifier
            driver_number: Filter by driver

        Returns:
            List of pit stop data
        """
        params = {'session_key': session_key}
        if driver_number:
            params['driver_number'] = driver_number

        return self._make_request('/pit', params)

    def get_stints(
        self,
        session_key: int,
        driver_number: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get tire stint data

        Args:
            session_key: Session identifier
            driver_number: Filter by driver

        Returns:
            List of stint data with compounds and ages
        """
        params = {'session_key': session_key}
        if driver_number:
            params['driver_number'] = driver_number

        return self._make_request('/stints', params)

    def get_overtakes(
        self,
        session_key: int,
        overtaking_driver_number: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get overtake data

        Args:
            session_key: Session identifier
            overtaking_driver_number: Filter by overtaking driver

        Returns:
            List of overtake events
        """
        params = {'session_key': session_key}
        if overtaking_driver_number:
            params['overtaking_driver_number'] = overtaking_driver_number

        return self._make_request('/overtakes', params)

    def get_positions(
        self,
        session_key: int,
        driver_number: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get position tracking data (real-time)

        Args:
            session_key: Session identifier
            driver_number: Filter by driver

        Returns:
            List of position changes with timestamps
        """
        params = {'session_key': session_key}
        if driver_number:
            params['driver_number'] = driver_number

        return self._make_request('/position', params)

    def get_intervals(
        self,
        session_key: int,
        driver_number: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get interval data (gaps between drivers)

        Args:
            session_key: Session identifier
            driver_number: Filter by driver

        Returns:
            List of interval data
        """
        params = {'session_key': session_key}
        if driver_number:
            params['driver_number'] = driver_number

        return self._make_request('/intervals', params)

    def get_championship_drivers(
        self,
        session_key: Optional[int] = None,
        driver_number: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get driver championship standings

        Args:
            session_key: Filter by session
            driver_number: Filter by driver

        Returns:
            List of championship standings
        """
        params = {}
        if session_key:
            params['session_key'] = session_key
        if driver_number:
            params['driver_number'] = driver_number

        return self._make_request('/championship_drivers', params)

    def get_championship_teams(
        self,
        session_key: Optional[int] = None,
        team_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get team championship standings

        Args:
            session_key: Filter by session
            team_name: Filter by team

        Returns:
            List of team standings
        """
        params = {}
        if session_key:
            params['session_key'] = session_key
        if team_name:
            params['team_name'] = team_name

        return self._make_request('/championship_teams', params)

    def get_starting_grid(
        self,
        session_key: int
    ) -> List[Dict[str, Any]]:
        """
        Get starting grid positions

        Args:
            session_key: Session identifier

        Returns:
            List of grid positions
        """
        params = {'session_key': session_key}
        return self._make_request('/starting_grid', params)

    def collect_race_weekend(
        self,
        meeting_key: int,
        include_practice: bool = False
    ) -> Dict[str, Any]:
        """
        Collect complete race weekend data

        Args:
            meeting_key: Meeting identifier
            include_practice: Include practice sessions

        Returns:
            Complete weekend data
        """
        logger.info(f"Collecting weekend data for meeting {meeting_key}")

        # Get all sessions for this meeting
        sessions = self.get_sessions(meeting_key=meeting_key)

        if not sessions:
            logger.warning(f"No sessions found for meeting {meeting_key}")
            return {}

        weekend_data = {
            'meeting_key': meeting_key,
            'sessions': {}
        }

        for session in tqdm(sessions, desc="Processing sessions"):
            session_key = session['session_key']
            session_name = session['session_name']

            # Skip practice if not requested
            if not include_practice and 'Practice' in session_name:
                continue

            session_data = self._collect_session_data(session_key, session_name)
            weekend_data['sessions'][session_name] = session_data

        return weekend_data

    def _collect_session_data(
        self,
        session_key: int,
        session_name: str
    ) -> Dict[str, Any]:
        """Collect complete data for a single session"""

        logger.debug(f"Collecting {session_name} data (session_key: {session_key})")

        session_data = {
            'session_key': session_key,
            'session_name': session_name,
            'results': self.get_session_results(session_key),
            'laps': self.get_laps(session_key),
            'pit_stops': self.get_pit_stops(session_key),
            'stints': self.get_stints(session_key),
            'drivers': self.get_drivers(session_key)
        }

        # Race-specific data
        if session_name == 'Race':
            session_data['overtakes'] = self.get_overtakes(session_key)
            session_data['starting_grid'] = self.get_starting_grid(session_key)

        return session_data

    def collect_season_overview(
        self,
        year: int
    ) -> Dict[str, Any]:
        """
        Collect season overview data

        Args:
            year: Season year

        Returns:
            Season data with meetings and championship
        """
        logger.info(f"Collecting season {year} overview")

        # Get all meetings
        meetings = self.get_meetings(year=year)

        # Get latest championship standings
        sessions = self.get_sessions(year=year, session_name='Race')
        latest_session = max(sessions, key=lambda x: x['date_end']) if sessions else None

        season_data = {
            'year': year,
            'meetings': meetings,
            'driver_championship': [],
            'team_championship': []
        }

        if latest_session:
            session_key = latest_session['session_key']
            season_data['driver_championship'] = self.get_championship_drivers(session_key)
            season_data['team_championship'] = self.get_championship_teams(session_key)

        return season_data


def main():
    """Example usage"""
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='OpenF1 data collector')
    parser.add_argument('--year', type=int, default=2024, help='Season year')
    parser.add_argument('--meeting', type=int, help='Specific meeting key')
    parser.add_argument('--session', type=int, help='Specific session key')

    args = parser.parse_args()

    collector = OpenF1Collector()

    if args.session:
        # Collect specific session
        print(f"\nCollecting session {args.session}...")
        data = collector._collect_session_data(args.session, "Session")
        print(f"\nResults: {len(data['results'])} drivers")
        print(f"Laps: {len(data['laps'])} laps")
        print(f"Pit stops: {len(data['pit_stops'])} stops")

    elif args.meeting:
        # Collect race weekend
        print(f"\nCollecting meeting {args.meeting}...")
        data = collector.collect_race_weekend(args.meeting)
        print(f"\nSessions: {list(data['sessions'].keys())}")

    else:
        # Season overview
        print(f"\nCollecting {args.year} season overview...")
        data = collector.collect_season_overview(args.year)
        print(f"\nMeetings: {len(data['meetings'])}")
        print(f"Driver championship: {len(data['driver_championship'])} drivers")
        print(f"Team championship: {len(data['team_championship'])} teams")


if __name__ == "__main__":
    main()
