"""
Feature engineering for FPL Selector.

This module transforms raw gameweek data into ML-ready features.
"""

import logging
from typing import List, Optional

import polars as pl
import fpl_selector.data.storage as storage

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for FPL data.
    
    Transforms raw gameweek data into features suitable for ML models:
    - Rolling statistics (form over N gameweeks)
    - Fixture difficulty adjustments
    - Team-level aggregations
    - Player momentum features
    """
    
    # Default rolling windows for form calculation
    DEFAULT_WINDOWS = [3, 5, 10]
    
    # Core stats to compute rolling averages for
    ROLLING_STATS = [
        "total_points",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "bps",
        "influence",
        "creativity",
        "threat",
    ]
    
    def __init__(
        self,
        rolling_windows: Optional[List[int]] = None,
        include_opponent_features: bool = True,
    ):
        """
        Initialize feature engineer.
        
        Args:
            rolling_windows: List of window sizes for rolling stats
            include_opponent_features: Whether to add opponent strength features
        """
        self.rolling_windows = rolling_windows or self.DEFAULT_WINDOWS
        self.include_opponent_features = include_opponent_features
        self.storage = storage.DataStorage()
        self.logger = logging.getLogger(self.__class__.__name__)

    def _build_teams_df(self, df: pl.DataFrame) -> pl.DataFrame:
        """Build a DataFrame of unique teams and their attributes."""
        if "team" not in df.columns:
            self.logger.warning("No 'team' column found in DataFrame.")
            return pl.DataFrame()
        
        teams_df = df.select(["team"]).unique()
        teams_df = teams_df.join(self.storage.load_teams(), left_on="team", how="left", right_on="name")
        return teams_df
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply all feature engineering transformations.
        
        Args:
            df: Raw gameweek DataFrame
        
        Returns:
            DataFrame with engineered features
        """
        self.logger.info(f"Engineering features for {len(df)} rows")
        
        # Ensure chronological order
        df = self._sort_chronological(df)
        
        # Add rolling statistics
        for window in self.rolling_windows:
            df = self._add_rolling_stats(df, window)
        
        # Add form trend
        df = self._add_form_trend(df)
        
        # Add minutes-based features
        df = self._add_minutes_features(df)
        
        # Add value-based features
        df = self._add_value_features(df)
        
        # Add team performance context
        df = self._add_team_performance(df)
        
        # Add fixture features if opponent data available
        if self.include_opponent_features and "opponent_team" in df.columns:
            df = self._add_opponent_features(df)
        
        self.logger.info(f"Generated {len(df.columns)} features")
        return df
    
    def _sort_chronological(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ensure data is sorted by player and time."""
        sort_cols = ["name"]
        if "kickoff_time" in df.columns:
            sort_cols.append("kickoff_time")
        elif "GW" in df.columns:
            sort_cols.append("GW")
        elif "round" in df.columns:
            sort_cols.append("round")
        
        return df.sort(sort_cols)
    
    def _add_rolling_stats(self, df: pl.DataFrame, window: int) -> pl.DataFrame:
        """Add rolling mean statistics for key metrics."""
        for stat in self.ROLLING_STATS:
            if stat not in df.columns:
                continue
            
            col_name = f"{stat}_roll_{window}"
            
            # Calculate rolling mean per player
            df = df.with_columns(
                pl.col(stat)
                .rolling_mean(window_size=window, min_periods=1)
                .over("name")
                .alias(col_name)
            )
        
        return df
    
    def _add_form_trend(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add form trend (recent form vs longer-term form)."""
        if "total_points" not in df.columns:
            return df
        
        # Short-term form (3 GW)
        if "total_points_roll_3" not in df.columns:
            df = self._add_rolling_stats(df, 3)
        
        # Long-term form (10 GW)
        if "total_points_roll_10" not in df.columns:
            df = self._add_rolling_stats(df, 10)
        
        # Trend: positive means improving form
        df = df.with_columns(
            (pl.col("total_points_roll_3") - pl.col("total_points_roll_10"))
            .alias("form_trend")
        )
        
        return df
    
    def _add_minutes_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add features related to playing time."""
        if "minutes" not in df.columns:
            return df
        
        # Points per 90 minutes (with safety for low minutes)
        if "total_points" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("minutes") >= 45)
                .then(pl.col("total_points") / (pl.col("minutes") / 90))
                .otherwise(None)
                .alias("points_per_90")
            )
        
        # Rolling minutes (availability proxy)
        df = df.with_columns(
            pl.col("minutes")
            .rolling_mean(window_size=5, min_periods=1)
            .over("name")
            .alias("minutes_roll_5")
        )
        
        # Started flag
        if "starts" not in df.columns:
            df = df.with_columns(
                (pl.col("minutes") >= 60).cast(pl.Int8).alias("likely_starter")
            )
        
        return df
    
    def _add_value_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add value-based features."""
        if "value" not in df.columns:
            return df
        
        # Points per million
        if "total_points" in df.columns:
            df = df.with_columns(
                (pl.col("total_points") / (pl.col("value") / 10))
                .alias("points_per_million")
            )
        
        # Value change (momentum in the transfer market)
        df = df.with_columns(
            pl.col("value")
            .diff()
            .over("name")
            .alias("value_change")
        )
        
        return df
    
    def _add_team_performance(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add team-level performance context."""
        if "team" not in df.columns:
            return df
        
        gw_col = "GW" if "GW" in df.columns else "round"
        if gw_col not in df.columns:
            return df
        
        # Team goals per gameweek
        if "goals_scored" in df.columns:
            team_goals = (
                df.group_by(["team", gw_col])
                .agg(pl.col("goals_scored").sum().alias("team_goals"))
            )
            df = df.join(team_goals, on=["team", gw_col], how="left")
        
        return df
    
    def _add_fixture_difficulty(self, df: pl.DataFrame) -> pl.DataFrame:
        """" Add features to compute difficulty of the upcoming feature"""
        if "opponent_team" not in df.columns:
            self.logger.warning("No 'opponent_team' column found for fixture difficulty.")
            return df

        # opp_strength = df.select(["opponent_team"])
        # .unique().join(self.storage.load_teams(), left_on="opponent_team", right_on="name", how="left")
        pass
    
    def _add_opponent_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add features based on opponent strength."""
        # Calculate team defensive strength (goals conceded)
        gw_col = "GW" if "GW" in df.columns else "round"
        
        if "goals_conceded" in df.columns and gw_col in df.columns:
            # Average goals conceded by each team
            team_defense = (
                df.group_by("team")
                .agg(pl.col("goals_conceded").mean().alias("team_avg_conceded"))
            )
            
            # Map opponent strength
            # This is a simplified version - in reality we'd use a lookup
            df = df.with_columns(
                pl.col("opponent_team").cast(pl.Utf8).alias("opponent_str")
            )
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names generated by this engineer."""
        features = []
        
        # Rolling features
        for window in self.rolling_windows:
            for stat in self.ROLLING_STATS:
                features.append(f"{stat}_roll_{window}")
        
        # Derived features
        features.extend([
            "form_trend",
            "points_per_90",
            "minutes_roll_5",
            "likely_starter",
            "points_per_million",
            "value_change",
            "team_goals",
        ])
        
        return features
