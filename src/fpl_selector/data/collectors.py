"""
Data collectors for FPL Selector.

This module provides collectors for fetching data from:
- Vaastav's Fantasy-Premier-League GitHub repository (historical gameweek data)
- Official FPL API (current player info, fixtures, user teams)
"""

import asyncio
import datetime
import io
import logging
import time
from functools import wraps
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import polars as pl
from aiohttp import ClientSession, ClientTimeout, ClientError

logger = logging.getLogger(__name__)


@dataclass
class CollectorConfig:
    """Configuration for data collectors."""
    vaastav_base_url: str = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
    fpl_api_base_url: str = "https://fantasy.premierleague.com/api"
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

# Implement rate limit mechanism
def rate_limit(calls_per_minute=10.0):
    print(f'Type of calls/min: {type(calls_per_minute)}, {calls_per_minute}')
    min_interval = 1.0 / calls_per_minute
    min_interval *= 60
    last_call = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = (time.time() * 60) - last_call
            print(f'Elapsed: {elapsed} vs. {min_interval}')
            if (elapsed * 60) - last_call < min_interval:
                time.sleep()
            last_call[0] = time.time() * 60
            return func(*args, **kwargs)
        return wrapper
    return decorator

class BaseCollector(ABC):
    """Abstract base class for data collectors."""
    
    def __init__(self, config: Optional[CollectorConfig] = None):
        self.config = config or CollectorConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._semaphore = asyncio.Semaphore(self.config.max_retries)  # For rate limiting
    
    @abstractmethod
    async def collect(self, **kwargs) -> pl.DataFrame:
        """Collect data and return as a Polars DataFrame."""
        pass
    
    async def _fetch_with_retry(
        self, 
        session: ClientSession, 
        url: str,
        as_json: bool = False
    ) -> Optional[str | Dict]:
        """Fetch URL with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        if as_json:
                            return await response.json()
                        return await response.text()
                    elif response.status == 404:
                        self.logger.warning(f"Resource not found: {url}")
                        return None
                    else:
                        self.logger.warning(
                            f"HTTP {response.status} for {url}, attempt {attempt + 1}"
                        )
            except ClientError as e:
                self.logger.warning(f"Client error for {url}: {e}, attempt {attempt + 1}")
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout for {url}, attempt {attempt + 1}")
            
            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay_seconds * (attempt + 1))
        
        self.logger.error(f"Failed to fetch {url} after {self.config.max_retries} attempts")
        return None


class VastaavCollector(BaseCollector):
    """
    Collector for Vaastav's Fantasy-Premier-League dataset.
    
    Data source: https://github.com/vaastav/Fantasy-Premier-League
    
    This dataset contains historical gameweek data updated weekly after each
    Premier League matchday.
    """

    def __init__(self, config: Optional[CollectorConfig] = None):
        super().__init__(config)
        self._current_season = self._calculate_current_season()

    def _calculate_current_season(self) -> str:
        """Calculate the current FPL season string (e.g., '2025-26')."""
        now = datetime.datetime.now()
        # FPL season starts in August
        if now.month >= 8:
            return f"{now.year}-{str(now.year + 1)[-2:]}"
        else:
            return f"{now.year - 1}-{str(now.year)[-2:]}"

    @property
    def current_season(self) -> str:
        """Get the current season string."""
        return self._current_season

    async def collect(
        self,
        season: Optional[str] = None,
        gameweeks: Optional[List[int]] = None,
    ) -> pl.DataFrame:
        """
        Collect gameweek data from Vaastav dataset.
        
        Args:
            season: Season string (e.g., '2024-25'). Defaults to current season.
            gameweeks: Specific gameweeks to fetch. If None, fetches merged_gw.csv.
        
        Returns:
            Polars DataFrame with gameweek data.
        """
        season = season or self._current_season
        timeout = ClientTimeout(total=self.config.timeout_seconds)
        async with ClientSession(timeout=timeout) as session:
            if gameweeks:
                return await self._collect_specific_gameweeks(session, season, gameweeks)
            else:
                return await self._collect_merged_gameweeks(session, season)

    async def _collect_merged_gameweeks(
        self, 
        session: ClientSession, 
        season: str
    ) -> pl.DataFrame:
        """Fetch the merged gameweek file containing all gameweeks."""
        url = f"{self.config.vaastav_base_url}/{season}/gws/merged_gw.csv"
        self.logger.info(f"Fetching merged gameweeks for {season}")

        data = await self._fetch_with_retry(session, url)
        if data is None:
            raise ValueError(f"Failed to fetch merged gameweeks for season {season}")

        df = pl.read_csv(io.StringIO(data))
        df = df.with_columns(pl.col(pl.Int64, pl.Float64).shrink_dtype())
        df = df.with_columns(
            pl.lit(season).alias("season"),
            pl.col("kickoff_time").str.to_datetime(),
            (pl.col("value") / 10)
        )

        self.logger.info(f"Collected {len(df)} rows for season {season}")
        return df

    async def _collect_specific_gameweeks(
        self,
        session: ClientSession,
        season: str,
        gameweeks: List[int],
    ) -> pl.DataFrame:
        """Fetch specific gameweek files."""
        base_url = f"{self.config.vaastav_base_url}/{season}/gws"

        tasks = [
            self._fetch_gameweek(session, base_url, gw, season)
            for gw in gameweeks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        dfs = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.warning(f"Failed to fetch GW {gameweeks[i]}: {result}")
            elif result is not None:
                dfs.append(result)

        if not dfs:
            raise ValueError(f"Failed to fetch any gameweeks: {gameweeks}")

        combined = pl.concat(dfs)
        self.logger.info(f"Collected {len(combined)} rows from {len(dfs)} gameweeks")
        return combined

    async def _fetch_gameweek(
        self,
        session: ClientSession,
        base_url: str,
        gw: int,
        season: str,
    ) -> Optional[pl.DataFrame]:
        """Fetch a single gameweek file."""
        url = f"{base_url}/gw{gw}.csv"
        data = await self._fetch_with_retry(session, url)

        if data is None:
            return None

        df = pl.read_csv(io.StringIO(data))
        df = df.with_columns(pl.col(pl.Int64, pl.Float64).shrink_dtype())
        df = df.with_columns([
            pl.lit(gw).alias("gw"),
            pl.lit(season).alias("season"),
            pl.col("kickoff_time").str.to_datetime(),
            # I don't think we need the extra precision
            # Player values range between 1e-1 to 1e-3 in precision
            (pl.col("value") / 10)
        ])
        return df

    async def collect_teams(self, season: Optional[str] = None) -> pl.DataFrame:
        """Fetch team information for a season."""
        season = season or self._current_season
        url = f"{self.config.vaastav_base_url}/{season}/teams.csv"

        timeout = ClientTimeout(total=self.config.timeout_seconds)
        async with ClientSession(timeout=timeout) as session:
            data = await self._fetch_with_retry(session, url)
            if data is None:
                raise ValueError(f"Failed to fetch teams for season {season}")

            return pl.read_csv(io.StringIO(data))

    async def collect_players(self, season: Optional[str] = None) -> pl.DataFrame:
        """Fetch player information for a season."""
        season = season or self._current_season
        url = f"{self.config.vaastav_base_url}/{season}/players_raw.csv"

        timeout = ClientTimeout(total=self.config.timeout_seconds)
        async with ClientSession(timeout=timeout) as session:
            data = await self._fetch_with_retry(session, url)
            if data is None:
                raise ValueError(f"Failed to fetch players for season {season}")

            return pl.read_csv(io.StringIO(data))


class FPLAPICollector(BaseCollector):
    """
    Collector for the official Fantasy Premier League API.
    
    API Base: https://fantasy.premierleague.com/api/
    
    Key endpoints:
    - /bootstrap-static/ - All static game data (players, teams, gameweeks)
    - /fixtures/ - All fixtures
    - /element-summary/{player_id}/ - Detailed player history
    - /entry/{team_id}/ - User's team info
    - /entry/{team_id}/event/{gw}/picks/ - User's picks for a gameweek
    """

    async def collect(self, **kwargs) -> pl.DataFrame:
        """
        Collect bootstrap data from FPL API.
        
        Returns combined players DataFrame with current season info.
        """
        bootstrap = await self.get_bootstrap_data()
        return self._parse_players(bootstrap)

    # @rate_limit(calls_per_minute=10)
    async def get_bootstrap_data(self) -> Dict[str, Any]:
        """
        Fetch bootstrap-static data from FPL API.
        
        This contains:
        - elements: All players with current stats
        - teams: All teams
        - events: All gameweeks
        - element_types: Position types
        """
        url = f"{self.config.fpl_api_base_url}/bootstrap-static/"

        timeout = ClientTimeout(total=self.config.timeout_seconds)
        async with ClientSession(timeout=timeout) as session:
            data = await self._fetch_with_retry(session, url, as_json=True)
            if data is None:
                raise ValueError("Failed to fetch bootstrap data from FPL API")
            return data

    def _parse_players(self, bootstrap: Dict[str, Any]) -> pl.DataFrame:
        """Parse player data from bootstrap response."""
        elements = bootstrap.get("elements", [])
        if not elements:
            raise ValueError("No player data in bootstrap response")

        df = pl.DataFrame(elements)

        # Add team names
        teams = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}
        # Used 'replace_strict' to ensure that data type conversion allows mapping from
        # integral ID to categorical ID
        df = df.with_columns(
            pl.col("team").replace_strict(teams).alias("team_name")
        )

        # Add position names
        positions = {p["id"]: p["singular_name_short"] for p in bootstrap.get("element_types", [])}
        df = df.with_columns(
            pl.col("element_type").replace_strict(positions).alias("position")
        )

        return df

    async def get_live_gameweek_data(self, gameweek: int) -> pl.DataFrame:
        """
        Fetch live gameweek data normalized to Vaastav schema.

        Returns DataFrame with columns matching merged_gw.csv format.
        """
        # Get live stats
        live_url = f"{self.config.fpl_api_base_url}/event/{gameweek}/live/"
        bootstrap = await self.get_bootstrap_data()
        fixture_df = await self.get_fixtures()
        timeout = ClientTimeout(total=self.config.timeout_seconds)
        async with ClientSession(timeout=timeout) as session:
            live_data = await self._fetch_with_retry(session, live_url, as_json=True)
            if live_data is None:
                raise ValueError(f"Failed to fetch live data for GW {gameweek}")

        players = {p["id"]: p for p in bootstrap.get("elements", [])}
        teams = {t["id"]: t["name"] for t in bootstrap.get("teams", [])}
        positions = {p["id"]: p["singular_name_short"] for p in bootstrap.get("element_types", [])}
        

        rows = []
        for elem in live_data.get("elements", []):
            player_id = elem['id']
            stats = elem.get('stats', {})
            player_info = players.get(player_id, {})

            # Map to Vaastav-compatible schema
            row = {
                # Player identification
                # "element": player_id,
                "name": player_info.get("web_name", ""),
                "position": positions.get(player_info.get("element_type"), ""),
                "team": teams.get(player_info.get("team"), ""),
                # Core stats (matching Vaastav column names)
                "total_points": stats.get("total_points", 0),
                "minutes": stats.get("minutes", 0),
                "goals_scored": stats.get("goals_scored", 0),
                "assists": stats.get("assists", 0),
                "clean_sheets": stats.get("clean_sheets", 0),
                "goals_conceded": stats.get("goals_conceded", 0),
                "own_goals": stats.get("own_goals", 0),
                "penalties_saved": stats.get("penalties_saved", 0),
                "penalties_missed": stats.get("penalties_missed", 0),
                "yellow_cards": stats.get("yellow_cards", 0),
                "red_cards": stats.get("red_cards", 0),
                "saves": stats.get("saves", 0),
                "bonus": stats.get("bonus", 0),
                "bps": stats.get("bps", 0),
                # Ownership and transfers (VERIFY EXISTENCE OF THESE FIELDS)
                "tackles": stats.get("tackles", 0),
                "recoveries": stats.get("recoveries", 0),
                "clearances_blocks_interceptions": stats.get("clearances_blocks_interceptions", 0),
                "defensive_contribution": stats.get("defensive_contribution", 0),
                "starts": stats.get("starts", 0),
                "selected": stats.get("selected", 0),
                "transfers_in": stats.get("transfers_in", 0),
                "transfers_out": stats.get("transfers_out", 0),
                "transfers_balance": stats.get("transfers_balance", 0),

                # Expected stats
                "expected_goals": float(stats.get("expected_goals", 0)),
                "expected_assists": float(stats.get("expected_assists", 0)),
                "expected_goal_involvements": float(
                    stats.get("expected_goal_involvements", 0)
                ),
                "expected_goals_conceded": float(
                    stats.get("expected_goals_conceded", 0)
                ),
                # ICT Index
                "influence": float(stats.get("influence", 0)),
                "creativity": float(stats.get("creativity", 0)),
                "threat": float(stats.get("threat", 0)),
                "ict_index": float(stats.get("ict_index", 0)),
                # Value (from bootstrap, not live)
                "value": player_info.get("now_cost", 0) / 10,
                # Gameweek identifier
                "GW": gameweek,
            }

            rows.append(row)
        return pl.DataFrame(rows)

    async def get_fixtures(self, gameweek: int | None = None) -> pl.DataFrame:
        """Fetch all fixtures for the current season."""
        url = f"{self.config.fpl_api_base_url}/fixtures/"

        timeout = ClientTimeout(total=self.config.timeout_seconds)
        async with ClientSession(timeout=timeout) as session:
            if gameweek:
                url += f'?event={gameweek}'
            data = await self._fetch_with_retry(session, url, as_json=True)
            if data is None:
                raise ValueError("Failed to fetch fixtures from FPL API")

            return pl.DataFrame(data)

    async def get_player_history(self, player_id: int) -> Dict[str, Any]:
        """Fetch detailed history for a specific player."""
        url = f"{self.config.fpl_api_base_url}/element-summary/{player_id}/"

        timeout = ClientTimeout(total=self.config.timeout_seconds)
        async with ClientSession(timeout=timeout) as session:
            data = await self._fetch_with_retry(session, url, as_json=True)
            if data is None:
                raise ValueError(f"Failed to fetch history for player {player_id}")
            return data

    async def get_user_team(self, team_id: int) -> Dict[str, Any]:
        """Fetch a user's team information."""
        url = f"{self.config.fpl_api_base_url}/entry/{team_id}/"

        timeout = ClientTimeout(total=self.config.timeout_seconds)
        async with ClientSession(timeout=timeout) as session:
            data = await self._fetch_with_retry(session, url, as_json=True)
            if data is None:
                raise ValueError(f"Failed to fetch team {team_id}")
            return data

    async def get_user_picks(self, team_id: int, gameweek: int) -> Dict[str, Any]:
        """Fetch a user's picks for a specific gameweek."""
        url = f"{self.config.fpl_api_base_url}/entry/{team_id}/event/{gameweek}/picks/"

        timeout = ClientTimeout(total=self.config.timeout_seconds)
        async with ClientSession(timeout=timeout) as session:
            data = await self._fetch_with_retry(session, url, as_json=True)
            if data is None:
                raise ValueError(f"Failed to fetch picks for team {team_id}, GW {gameweek}")
            return data

    async def get_current_gameweek(self) -> int:
        """Get the current gameweek number."""
        bootstrap = await self.get_bootstrap_data()
        events = bootstrap.get("events", [])

        for event in events:
            if event.get("is_current"):
                return event["id"]

        # If no current gameweek (between seasons), return the last finished one
        for event in reversed(events):
            if event.get("finished"):
                return event["id"]

        return 1


# Convenience function for synchronous usage
def collect_gameweek_data(
    season: Optional[str] = None,
    gameweeks: Optional[List[int]] = None,
    config: Optional[CollectorConfig] = None,
) -> pl.DataFrame:
    """
    Synchronous wrapper for collecting gameweek data.
    
    Args:
        season: Season string (e.g., '2024-25')
        gameweeks: Specific gameweeks to fetch
        config: Collector configuration
    
    Returns:
        Polars DataFrame with gameweek data
    """
    collector = VastaavCollector(config)
    return asyncio.run(collector.collect(season=season, gameweeks=gameweeks))


def collect_current_players(config: Optional[CollectorConfig] = None) -> pl.DataFrame:
    """
    Synchronous wrapper for collecting current player data.
    
    Returns:
        Polars DataFrame with current player information
    """
    collector = FPLAPICollector(config)
    return asyncio.run(collector.collect())
