from time import time
import pandas as pd
import numpy as np
from typing import List, Dict, Union
from sklearn.base import BaseEstimator, TransformerMixin
from utils import Database


class Timer:
    def __init__(self):
        self.start = time()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.end = time()
        self.interval = self.end - self.start
        print(f"Time taken: {self.interval:.2f} seconds")


class FPLFeatureEngineering(BaseEstimator, TransformerMixin):
    """Feature engineering pipeline for FPL data"""
    def __init__(self, 
                 rolling_windows: List[int] = [3, 5, 10],
                 form_window_days: int = 30,
                 include_opponent_stats: bool = True):
        self.rolling_windows = rolling_windows
        self.form_window_days = form_window_days
        self.include_opponent_stats = include_opponent_stats
        self.db = Database()

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['kickoff_time'] = pd.to_datetime(df.kickoff_time, utc=True)
        df = self._sort_chronological(df)

        # Basic rolling statistics
        for window in self.rolling_windows:
            df = self._add_rolling_stats(df, window)

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
                )

                # Add standard deviation for key metrics
                df[f'{stat}_rolling_{window}_std'] = (
                    df.groupby('name')[stat]
                    .rolling(window, min_periods=1)
                    .std()
                    .reset_index(0, drop=True)
                )

        return df

    def _calculate_form(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate form using time-based window"""
        form_query = f"""
        WITH player_form AS (
            SELECT
                name,
                kickoff_time,
                --total_points,
                CAST(AVG(total_points) OVER (
                    PARTITION BY name
                    ORDER BY kickoff_time
                    RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
                ) AS DECIMAL(3, 1)) AS form
            FROM gameweek_stats
        )
        SELECT pf.* FROM player_form pf
        JOIN gameweek_stats gs
        ON pf.name = gs.name
        AND pf.kickoff_time = gs.kickoff_time
        WHERE gs.season = :season;"""
        def compute_form(group):
            window = f'{self.form_window_days}D'
            return (
                group.set_index('kickoff_time')['total_points']
                .rolling(window, min_periods=1)
                .mean()
                .round(1)
                .reset_index(name='form')
            )
        print(f'SEASON: {df.season.iloc[0]}')
        with Timer() as t1:
            form_df = self.db.get_data(form_query, params={'season': df.season.iloc[0]})

        form_df['kickoff_time'] = pd.to_datetime(form_df.kickoff_time, utc=True)
        # print(df.kickoff_time.dtype, form_df.kickoff_time.dtype)
        # df["form"] = pd.merge(
        #     form_df, df, on=["name", "kickoff_time"], how="inner"
        # )
        df = pd.merge(form_df, df, on=["name", "kickoff_time"], how="inner")
        # print(form_df.head())
        with Timer() as t2:
            form_df = df.groupby('name').apply(compute_form).reset_index()

        # form_df = df.groupby('name').apply(compute_form).reset_index()
        print(f'Time taken to get form data: {t1.interval:.2f} seconds')
        print(f'Time taken to calculate form: {t2.interval:.2f} seconds')
        # print(f'Form DF: \n{form_df.head()}')
        # df['form'] = df.merge(form_df[['name', 'kickoff_time', 'form']],
        #   on=['name', 'kickoff_time'], how='left')['form']
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

        # Map these back to the fixtures
        df['opponent_strength'] = df['opponent_team'].map(team_strength['expected_goals'])

        return df

    def _add_team_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team performance context"""
        # Team form
        df['team_form'] = (
            df.groupby('team')['total_points']
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        # Team goals scoring form
        df['team_scoring_form'] = (
            df.groupby('team')['goals_scored']
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features"""
        # Consecutive starts
        def consecutive_starts(group):
            return (group > 60).astype(int).cumsum()
        res = df.groupby('name')['minutes'] \
            .apply(consecutive_starts)
        print(res.head())
        df['consecutive_starts'] = (
            df.groupby('name')['minutes']
            .apply(lambda x: (x > 60).astype(int).cumsum())
        )

        # Points trend (positive or negative momentum)
        df['points_trend'] = (
            df.groupby('name')['total_points']
            .diff()
            .rolling(3, min_periods=1)
            .mean()
        )

        return df
