"""
Data storage layer for FPL Selector.

This module provides a simple file-based storage system using Parquet files
for efficient columnar storage of FPL data. No external database required.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import polars as pl

logger = logging.getLogger(__name__)


class DataStorage:
    """
    File-based storage for FPL data using Parquet format.
    
    Directory Structure:
        data/
        ├── raw/
        │   ├── gameweeks/
        │   │   ├── 2024-25.parquet
        │   │   └── 2025-26.parquet
        │   ├── players/
        │   │   └── current.parquet
        │   ├── fixtures/
        │   │   └── current.parquet
        │   └── teams/
        │       └── 2025-26.parquet
        ├── processed/
        │   └── features/
        │       └── 2025-26.parquet
        └── models/
            ├── xgb_latest.joblib
            └── metadata.json
    """
    
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize storage with a base path.
        
        Args:
            base_path: Root directory for data storage. Defaults to 'data/' 
                      in the project root.
        """
        if base_path is None:
            # Default to data/ directory relative to project root
            base_path = Path(__file__).parent.parent.parent.parent / "data"
        
        self.base_path = Path(base_path)
        self._ensure_directories()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _ensure_directories(self) -> None:
        """Create required directory structure if it doesn't exist."""
        directories = [
            self.base_path / "raw" / "gameweeks",
            self.base_path / "raw" / "players",
            self.base_path / "raw" / "fixtures",
            self.base_path / "raw" / "teams",
            self.base_path / "processed" / "features",
            self.base_path / "models",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    # ========== GAMEWEEK DATA ==========
    
    def save_gameweeks(self, df: pl.DataFrame, season: str) -> Path:
        """
        Save gameweek data for a season.
        
        Args:
            df: DataFrame with gameweek data
            season: Season string (e.g., '2024-25')
        
        Returns:
            Path to saved file
        """
        path = self.base_path / "raw" / "gameweeks" / f"{season}.parquet"
        df.write_parquet(path)
        self.logger.info(f"Saved {len(df)} gameweek rows to {path}")
        return path
    
    def load_gameweeks(
        self, 
        season: Optional[str] = None,
        seasons: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        Load gameweek data for one or more seasons.
        
        Args:
            season: Single season to load
            seasons: List of seasons to load
        
        Returns:
            DataFrame with gameweek data
        """
        gameweeks_dir = self.base_path / "raw" / "gameweeks"
        
        if season:
            path = gameweeks_dir / f"{season}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"No gameweek data for season {season}")
            return pl.read_parquet(path)
        
        if seasons:
            dfs = []
            for s in seasons:
                path = gameweeks_dir / f"{s}.parquet"
                if path.exists():
                    dfs.append(pl.read_parquet(path))
                else:
                    self.logger.warning(f"No gameweek data for season {s}")
            
            if not dfs:
                raise FileNotFoundError(f"No gameweek data found for seasons: {seasons}")
            return pl.concat(dfs)
        
        # Load all available seasons
        parquet_files = list(gameweeks_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError("No gameweek data available")
        
        dfs = [pl.read_parquet(f) for f in sorted(parquet_files)]
        return pl.concat(dfs)
    
    def list_available_seasons(self) -> List[str]:
        """List all seasons with available gameweek data."""
        gameweeks_dir = self.base_path / "raw" / "gameweeks"
        return sorted([f.stem for f in gameweeks_dir.glob("*.parquet")])
    
    # ========== PLAYER DATA ==========
    
    def save_players(self, df: pl.DataFrame) -> Path:
        """
        Save current player data from FPL API.
        
        Args:
            df: DataFrame with player data
        
        Returns:
            Path to saved file
        """
        path = self.base_path / "raw" / "players" / "current.parquet"
        
        # Also save a timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.base_path / "raw" / "players" / f"backup_{timestamp}.parquet"
        
        df.write_parquet(path)
        df.write_parquet(backup_path)
        
        self.logger.info(f"Saved {len(df)} players to {path}")
        return path
    
    def load_players(self) -> pl.DataFrame:
        """Load current player data."""
        path = self.base_path / "raw" / "players" / "current.parquet"
        if not path.exists():
            raise FileNotFoundError("No player data available. Run 'fpl update' first.")
        return pl.read_parquet(path)
    
    # ========== FIXTURES DATA ==========
    
    def save_fixtures(self, df: pl.DataFrame) -> Path:
        """Save fixtures data."""
        path = self.base_path / "raw" / "fixtures" / "current.parquet"
        df.write_parquet(path)
        self.logger.info(f"Saved {len(df)} fixtures to {path}")
        return path
    
    def load_fixtures(self) -> pl.DataFrame:
        """Load fixtures data."""
        path = self.base_path / "raw" / "fixtures" / "current.parquet"
        if not path.exists():
            raise FileNotFoundError("No fixtures data available. Run 'fpl update' first.")
        return pl.read_parquet(path)
    
    # ========== TEAMS DATA ==========
    
    def save_teams(self, df: pl.DataFrame, season: str) -> Path:
        """Save team data for a season."""
        print(f'Saving {season} to parquet')
        path = self.base_path / "raw" / "teams" / f"{season}.parquet"
        df.write_parquet(path)
        self.logger.info(f"Saved {len(df)} teams to {path}")
        return path
    
    def load_teams(self, season: Optional[str] = None) -> pl.DataFrame:
        """Load team data."""
        teams_dir = self.base_path / "raw" / "teams"
        
        if season:
            path = teams_dir / f"{season}.parquet"
        else:
            # Load most recent
            files = sorted(teams_dir.glob("*.parquet"))
            if not files:
                raise FileNotFoundError("No teams data available")
            path = files[-1]
        
        if not path.exists():
            raise FileNotFoundError(f"No teams data for season {season}")
        return pl.read_parquet(path)
    
    # ========== PROCESSED FEATURES ==========
    
    def save_features(self, df: pl.DataFrame, season: str) -> Path:
        """Save feature-engineered data."""
        path = self.base_path / "processed" / "features" / f"{season}.parquet"
        df.write_parquet(path)
        self.logger.info(f"Saved {len(df)} feature rows to {path}")
        return path
    
    def load_features(
        self,
        season: Optional[str] = None,
        seasons: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """Load feature-engineered data."""
        features_dir = self.base_path / "processed" / "features"
        
        if season:
            path = features_dir / f"{season}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"No features for season {season}")
            return pl.read_parquet(path)
        
        if seasons:
            dfs = []
            for s in seasons:
                path = features_dir / f"{s}.parquet"
                if path.exists():
                    dfs.append(pl.read_parquet(path))
            if not dfs:
                raise FileNotFoundError(f"No features found for seasons: {seasons}")
            return pl.concat(dfs)
        
        # Load all
        files = list(features_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError("No feature data available")
        return pl.concat([pl.read_parquet(f) for f in sorted(files)])
    
    # ========== MODEL STORAGE ==========
    
    def get_model_path(self, model_name: str = "xgb_latest") -> Path:
        """Get path for a model file."""
        return self.base_path / "models" / f"{model_name}.joblib"
    
    def list_models(self) -> List[str]:
        """List available model files."""
        models_dir = self.base_path / "models"
        return [f.stem for f in models_dir.glob("*.joblib")]
    
    # ========== UTILITIES ==========
    
    def get_data_freshness(self) -> dict:
        """Get timestamps of when each data type was last updated."""
        freshness = {}
        
        paths = {
            "players": self.base_path / "raw" / "players" / "current.parquet",
            "fixtures": self.base_path / "raw" / "fixtures" / "current.parquet",
        }
        
        for name, path in paths.items():
            if path.exists():
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                freshness[name] = mtime
            else:
                freshness[name] = None
        
        # Check gameweeks
        gameweeks_dir = self.base_path / "raw" / "gameweeks"
        gw_files = list(gameweeks_dir.glob("*.parquet"))
        if gw_files:
            latest = max(f.stat().st_mtime for f in gw_files)
            freshness["gameweeks"] = datetime.fromtimestamp(latest)
        else:
            freshness["gameweeks"] = None
        
        return freshness
    
    def clear_cache(self, data_type: Optional[str] = None) -> None:
        """
        Clear cached data.
        
        Args:
            data_type: Specific type to clear ('players', 'fixtures', 'features')
                      If None, clears all non-gameweek data.
        """
        if data_type == "players":
            for f in (self.base_path / "raw" / "players").glob("*.parquet"):
                f.unlink()
        elif data_type == "fixtures":
            for f in (self.base_path / "raw" / "fixtures").glob("*.parquet"):
                f.unlink()
        elif data_type == "features":
            for f in (self.base_path / "processed" / "features").glob("*.parquet"):
                f.unlink()
        elif data_type is None:
            # Clear all except raw gameweeks (historical data)
            self.clear_cache("players")
            self.clear_cache("fixtures")
            self.clear_cache("features")
        
        self.logger.info(f"Cleared cache: {data_type or 'all'}")
