"""
Points prediction model for FPL Selector.

This module provides the XGBoost-based model for predicting player points.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import polars as pl
from sklearn import model_selection, metrics
import numpy as np

logger = logging.getLogger(__name__)


class PointsPredictor:
    """
    XGBoost-based model for predicting FPL player points.
    
    Trains separate models per position for better accuracy.
    """
    
    # Position-specific feature sets
    POSITION_FEATURES = {
        "GK": [
            "minutes_roll_5",
            "total_points_roll_3",
            "total_points_roll_5",
            "clean_sheets_roll_5",
            "bps_roll_5",
            "form_trend",
        ],
        "DEF": [
            "minutes_roll_5",
            "total_points_roll_3",
            "total_points_roll_5",
            "clean_sheets_roll_5",
            "goals_scored_roll_5",
            "assists_roll_5",
            "bps_roll_5",
            "threat_roll_5",
            "form_trend",
            "points_per_million",
        ],
        "MID": [
            "minutes_roll_5",
            "total_points_roll_3",
            "total_points_roll_5",
            "goals_scored_roll_5",
            "assists_roll_5",
            "expected_goals_roll_5",
            "expected_assists_roll_5",
            "bps_roll_5",
            "influence_roll_5",
            "creativity_roll_5",
            "threat_roll_5",
            "form_trend",
            "points_per_90",
            "points_per_million",
        ],
        "FWD": [
            "minutes_roll_5",
            "total_points_roll_3",
            "total_points_roll_5",
            "goals_scored_roll_5",
            "assists_roll_5",
            "expected_goals_roll_5",
            "expected_goal_involvements_roll_5",
            "bps_roll_5",
            "threat_roll_5",
            "form_trend",
            "points_per_90",
            "points_per_million",
        ],
    }
    
    TARGET = "total_points"
    
    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize predictor.
        
        Args:
            model_dir: Directory to save/load models from
        """
        self.model_dir = Path(model_dir) if model_dir else None
        self.models: Dict[str, Any] = {}  # Position -> trained model
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def train(
        self,
        df: pl.DataFrame,
        positions: Optional[List[str]] = None,
        **xgb_params,
    ) -> Dict[str, float]:
        """
        Train models for each position.
        
        Args:
            df: Feature-engineered DataFrame
            positions: Positions to train for (default: all)
            **xgb_params: Additional XGBoost parameters
        
        Returns:
            Dictionary of position -> validation RMSE
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")
        
        positions = positions or list(self.POSITION_FEATURES.keys())
        results = {}
        
        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": -1,
        }
        default_params.update(xgb_params)
        import sys
        for pos in positions[-1:]:
            self.logger.info(f"Training model for {pos}")
            
            # Filter position data
            pos_df = df.filter(pl.col("position") == pos)
            if len(pos_df) < 100:
                self.logger.warning(f"Insufficient data for {pos}: {len(pos_df)} rows")
                continue
            
            # Get features for this position
            features = self._get_available_features(pos_df, pos)
            if not features:
                self.logger.warning(f"No features available for {pos}")
                continue
            
            # Prepare data
            X = pos_df.select(features).to_pandas()
            print(pos, X.columns, sep='\n')
            y = pos_df.select(self.TARGET).to_pandas().values.ravel()
            
            # Handle missing values
            X = X.fillna(0)
            # Train/validation split (time-based)
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            print('X_Train', X_train, sep='\n')
            # sys.exit(0)
            
            # Train model
            model = xgb.XGBRegressor(**default_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            
            # Evaluate
            y_pred = model.predict(X_val)
            rmse = metrics.root_mean_squared_error(y_val, y_pred)
            # rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            
            self.models[pos] = model
            results[pos] = rmse
            self.logger.info(f"{pos} model RMSE: {rmse:.3f}")
        
        return results
    
    def predict(
        self,
        df: pl.DataFrame,
        positions: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        Generate predictions for all players.
        
        Args:
            df: Feature-engineered DataFrame
            positions: Positions to predict for
        
        Returns:
            DataFrame with added 'predicted_points' column
        """
        positions = positions or list(self.models.keys())
        predictions = []
        
        for pos in positions:
            if pos not in self.models:
                self.logger.warning(f"No model for position {pos}")
                continue
            
            model = self.models[pos]
            pos_df = df.filter(pl.col("position") == pos)
            
            if len(pos_df) == 0:
                continue
            
            features = self._get_available_features(pos_df, pos)
            X = pos_df.select(features).to_pandas().fillna(0)
            
            preds = model.predict(X)
            preds = np.floor(preds)
            pos_df = pos_df.with_columns(
                pl.Series("predicted_points", preds)
            )
            predictions.append(pos_df)
        
        if not predictions:
            return df.with_columns(pl.lit(None).alias("predicted_points"))
        
        return pl.concat(predictions)
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save all models to disk."""
        save_path = path or (self.model_dir / "models.joblib")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.models, save_path)
        self.logger.info(f"Saved models to {save_path}")
        return save_path
    
    def load(self, path: Optional[Path] = None) -> None:
        """Load models from disk."""
        load_path = path or (self.model_dir / "models.joblib")
        
        if not load_path.exists():
            raise FileNotFoundError(f"No models found at {load_path}")
        
        self.models = joblib.load(load_path)
        self.logger.info(f"Loaded models for positions: {list(self.models.keys())}")
    
    def _get_available_features(
        self,
        df: pl.DataFrame,
        position: str,
    ) -> List[str]:
        """Get features that exist in the DataFrame."""
        desired = self.POSITION_FEATURES.get(position, [])
        available = set(df.columns)
        return [f for f in desired if f in available]
    
    def get_feature_importance(self, position: str) -> Dict[str, float]:
        """Get feature importance for a position's model."""
        if position not in self.models:
            raise ValueError(f"No model for position {position}")
        
        model = self.models[position]
        features = self.POSITION_FEATURES.get(position, [])
        
        importance = dict(zip(features, model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
