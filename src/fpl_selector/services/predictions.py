"""
Prediction service for FPL Selector.

High-level interface for generating player predictions.
"""

import logging
from pathlib import Path
from typing import List, Optional

import polars as pl

from fpl_selector.data.storage import DataStorage
from fpl_selector.features.engineering import FeatureEngineer
from fpl_selector.models.predictor import PointsPredictor

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Service for generating and retrieving player predictions.
    
    Orchestrates data loading, feature engineering, and model inference.
    """
    
    def __init__(
        self,
        storage: Optional[DataStorage] = None,
        model_path: Optional[Path] = None,
    ):
        """
        Initialize prediction service.
        
        Args:
            storage: Data storage instance
            model_path: Path to trained model
        """
        self.storage = storage or DataStorage()
        self.feature_engineer = FeatureEngineer()
        self.predictor = PointsPredictor(model_dir=self.storage.base_path / "models")
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Try to load existing model
        try:
            self.predictor.load()
            self._model_loaded = True
        except FileNotFoundError:
            self._model_loaded = False
            self.logger.warning("No trained model found. Run training first.")
    
    def get_predictions(
        self,
        position: Optional[str] = None,
        min_minutes: int = 0,
        top_n: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Get predictions for all players.
        
        Args:
            position: Filter by position
            min_minutes: Minimum total minutes played
            top_n: Return only top N players
        
        Returns:
            DataFrame with predictions sorted by expected points
        """
        # Load current player data
        players = self.storage.load_players()
        
        if not self._model_loaded:
            # Fallback to form-based ranking
            self.logger.info("Using form as prediction proxy (no model trained)")
            players = players.with_columns(
                pl.col("form").cast(pl.Float64).alias("predicted_points")
            )
        else:
            # Load historical data for features
            try:
                gameweeks = self.storage.load_gameweeks()
                features = self.feature_engineer.transform(gameweeks)
                
                # Get latest features per player
                latest = features.group_by("name").tail(1)
                
                # Merge with current player data
                players = players.join(
                    latest.select(self.feature_engineer.get_feature_columns() + ["name"]),
                    left_on="web_name",
                    right_on="name",
                    how="left",
                )
                
                # Generate predictions
                players = self.predictor.predict(players)
                
            except FileNotFoundError:
                self.logger.warning("No gameweek data. Using form as proxy.")
                players = players.with_columns(
                    pl.col("form").cast(pl.Float64).alias("predicted_points")
                )
        
        # Apply filters
        if min_minutes > 0 and "minutes" in players.columns:
            players = players.filter(pl.col("minutes") >= min_minutes)
        
        if position:
            players = players.filter(pl.col("position") == position.upper())
        
        # Sort by predictions
        players = players.sort("predicted_points", descending=True)
        
        if top_n:
            players = players.head(top_n)
        
        return players
    
    def train_model(
        self,
        seasons: Optional[List[str]] = None,
        **xgb_params,
    ) -> dict:
        """
        Train the prediction model.
        
        Args:
            seasons: Seasons to use for training
            **xgb_params: XGBoost parameters
        
        Returns:
            Training results (RMSE per position)
        """
        # Load gameweek data
        if seasons:
            gameweeks = self.storage.load_gameweeks(seasons=seasons)
        else:
            gameweeks = self.storage.load_gameweeks()
        
        self.logger.info(f"Training on {len(gameweeks)} gameweek records")
        
        # Engineer features
        features = self.feature_engineer.transform(gameweeks)
        
        # Save features
        season = seasons[0] if seasons else "all"
        self.storage.save_features(features, season)
        
        # Train model
        results = self.predictor.train(features, **xgb_params)
        
        # Save model
        self.predictor.save()
        self._model_loaded = True
        
        return results
    
    def get_player_prediction(self, player_name: str) -> dict:
        """
        Get detailed prediction for a specific player.
        
        Args:
            player_name: Player name to look up
        
        Returns:
            Dictionary with prediction details
        """
        predictions = self.get_predictions()
        
        player = predictions.filter(
            pl.col("web_name").str.to_lowercase().str.contains(player_name.lower())
        )
        
        if len(player) == 0:
            return {"error": f"Player '{player_name}' not found"}
        
        row = player.head(1).to_dicts()[0]
        print(f'Player prediction for {row['name']}, value: {row['now_cost']}')
        
        return {
            "name": row.get("web_name"),
            "team": row.get("team_name"),
            "position": row.get("position"),
            "predicted_points": row.get("predicted_points"),
            "form": row.get("form"),
            "price": row.get("now_cost", 0) / 10,
            "selected_by": row.get("selected_by_percent"),
        }
