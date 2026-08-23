from time import time
import pandas as pd
import numpy as np
from typing import List, Dict, Union
from sklearn.base import BaseEstimator, TransformerMixin
from utils import Database, Timer


class FplPreprocessor(BaseEstimator, TransformerMixin):
    """Feature engineering pipeline for FPL data"""
    def __init__(self, 
                 lag_features: List[str],
                 rolling_windows: List[int] = [3, 5, 10],
                 form_window_days: int = 30,
                 include_opponent_stats: bool = True):
        self.lag_features = lag_features
        self.rolling_windows = rolling_windows
        self.form_window_days = form_window_days
        self.include_opponent_stats = include_opponent_stats
        self.db = Database()

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._sort_chronological(df)

        # Basic rolling statistics
        for window in self.rolling_windows:
            df = self._add_rolling_stats(df, window)

        # Lag features
        # df = self._add_lagged_features(df, self.lag_features)

        # Form calculations
        df = self._calculate_form(df)

        # Fixture difficulty
        df = self._add_fixture_difficulty(df)

        # Team performance metrics
        df = self._add_team_performance(df)

        # Player momentum features
        df = self._add_momentum_features(df)

        return df

    def _sort_chronological(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure data is in chronological order for each player"""
        return df.sort_values(['name', 'kickoff_time'])

    def _add_rolling_stats(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add rolling statistics for key metrics"""
        stats_to_roll = ['total_points', 'minutes', 'goals_scored', 
                        'assists', 'clean_sheets', 'expected_goals', 
                        'expected_assists']

        for stat in stats_to_roll:
            if stat in df.columns:
                df[f'{stat}_rolling_{window}'] = (
                    df.groupby('name')[stat]
                    .rolling(window, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                    # .dropna()
                )

                # Add standard deviation for key metrics
                # df[f'{stat}_rolling_{window}_std'] = (
                #     df.groupby('name')[stat]
                #     .rolling(window, min_periods=1)
                #     .std()
                #     .reset_index(0, drop=True)
                #     # .dropna()
                # )

        return df

    def _calculate_form(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate form using time-based window"""
        form_query = f"""
        WITH player_form AS (
            SELECT
                name,
                kickoff_time,
                CAST(AVG(total_points) OVER (
                    PARTITION BY name, season
                    ORDER BY kickoff_time
                    RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
                ) AS DECIMAL(3, 1)) AS form
            FROM gameweek_stats
        )
        SELECT pf.* FROM player_form pf
        JOIN gameweek_stats gs
        ON pf.name = gs.name
        AND pf.kickoff_time = gs.kickoff_time;"""

        with Timer() as t1:
            form_df = self.db.get_data(form_query)

        
        df = pd.merge(form_df, df, on=["name", "kickoff_time"], how="inner")

        return df

    def _add_fixture_difficulty(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fixture difficulty metrics"""
        # Calculate team strength metrics
        team_strength = (
            df.groupby('team')
            .agg({
                'goals_scored': 'mean',
                'goals_conceded': 'mean',
                'expected_goals': 'mean',
                'expected_goals_conceded': 'mean'
            })
            .rolling(5, min_periods=1)
            .mean()
        )
        print('----------TEAM STRENGTH INFO---------')
        print(team_strength.describe())
        print('----------TEAM STRENGTH INFO END---------')

        teams_df = self.db.get_data("""SELECT * FROM teams""")
        
        merged = df[['opponent_team', 'season']].merge(teams_df, left_on=['opponent_team', 'season'], right_on=['id', 'season'])
        df['opp_team_name'] = merged['name']
        
        # Map these back to the fixtures
        df['opponent_strength'] = df['opp_team_name'].map(team_strength['expected_goals'])
        
        return df

    def _add_team_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team performance context"""
        # Team form
        df['team_form'] = (
            df.groupby('team')['total_points']
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(drop=True)
        )

        # Team goals scoring form
        df['team_scoring_form'] = (
            df.groupby('team')['goals_scored']
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(drop=True)
        )

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features"""
        momentum_query = f"""
        SELECT
            name,
            kickoff_time,
            SUM(starts) OVER (
                PARTITION BY name
                ORDER BY kickoff_time
            ) AS consecutive_starts
        FROM gameweek_stats
        WHERE season = '{df.season.iloc[0]}';
        """
        momentum_df = self.db.get_data(momentum_query)
        df = pd.merge(momentum_df, df, on=['name', 'kickoff_time'])

        # Points trend (positive or negative momentum)
        df['points_trend'] = (
            df.groupby('name')['total_points']
            .diff()
            .rolling(3, min_periods=1)
            .mean().dropna()
        )

        return df

    def _add_lagged_features(self, df: pd.DataFrame, target_features: List[str]) -> pd.DataFrame:
        """Add lagged features for time series forecasting"""
        # print(f'LAGGED FEATURES METHOD: {target_features in df.columns}')
        print(f'params: {",".join([f":{feature}" for feature in target_features])}')
        query = """
        SELECT 
            name,
            kickoff_time,
            LAG(total_points, 1) OVER (
                PARTITION BY name, season
                ORDER BY kickoff_time
            ) AS lag_pts,
            ,
            gw, 
            season
        FROM gameweek_stats
        """
        # lag_df = self.db.get_data(query)

        lag_dataframe = df.copy()
        lag_dataframe['lag_pts'] = lag_dataframe.groupby(['name', 'season'])[target_features].shift(1)
        lag_dataframe = lag_dataframe.dropna(axis = 0) # Drop the old prediction
        return lag_dataframe
