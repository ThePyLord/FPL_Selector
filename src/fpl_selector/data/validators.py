"""
Data validation for FPL Selector.

This module provides validators to ensure data quality and consistency
before processing or model training.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def __bool__(self) -> bool:
        return self.is_valid
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = []
        status = "✓ VALID" if self.is_valid else "✗ INVALID"
        lines.append(f"Validation: {status}")
        
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            for err in self.errors[:5]:  # Show first 5
                lines.append(f"  - {err}")
            if len(self.errors) > 5:
                lines.append(f"  ... and {len(self.errors) - 5} more")
        
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for warn in self.warnings[:5]:
                lines.append(f"  - {warn}")
            if len(self.warnings) > 5:
                lines.append(f"  ... and {len(self.warnings) - 5} more")
        
        if self.stats:
            lines.append("Stats:")
            for key, value in self.stats.items():
                lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)


class DataValidator:
    """
    Validates FPL data for quality and consistency.
    
    Checks include:
    - Required columns presence
    - Data type correctness
    - Value range validation
    - Missing data thresholds
    - Logical consistency
    """
    
    # Expected columns for gameweek data
    GAMEWEEK_REQUIRED_COLUMNS = {
        "name",
        "position",
        "team",
        "total_points",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "bonus",
        "bps",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "value",
        "round",  # or GW
    }
    
    # Expected columns for player data from FPL API
    PLAYER_REQUIRED_COLUMNS = {
        "id",
        "web_name",
        "team",
        "element_type",
        "now_cost",
        "total_points",
        "form",
        "selected_by_percent",
        "minutes",
    }
    
    # Expected columns for fixtures
    FIXTURE_REQUIRED_COLUMNS = {
        "id",
        "event",
        "team_h",
        "team_a",
        "team_h_difficulty",
        "team_a_difficulty",
    }
    
    # Valid positions
    VALID_POSITIONS = {"GKP", "GK", "DEF", "MID", "FWD"}
    
    def __init__(self, strict: bool = False):
        """
        Initialize validator.
        
        Args:
            strict: If True, warnings are treated as errors.
        """
        self.strict = strict
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_gameweek_data(self, df: pl.DataFrame) -> ValidationResult:
        """
        Validate gameweek data from Vaastav dataset.
        
        Args:
            df: DataFrame with gameweek data
        
        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        stats = {}
        
        # Check required columns
        missing_cols = self.GAMEWEEK_REQUIRED_COLUMNS - set(df.columns)
        # Allow 'GW' instead of 'round'
        if "round" in missing_cols and "GW" in df.columns:
            missing_cols.discard("round")
        
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check for empty DataFrame
        if len(df) == 0:
            errors.append("DataFrame is empty")
            return ValidationResult(is_valid=False, errors=errors)
        
        stats["row_count"] = len(df)
        stats["unique_players"] = df["name"].n_unique() if "name" in df.columns else 0
        
        # Check positions
        if "position" in df.columns:
            positions = set(df["position"].unique().to_list())
            invalid_positions = positions - self.VALID_POSITIONS
            if invalid_positions:
                warnings.append(f"Unknown positions found: {invalid_positions}")
            stats["positions"] = list(positions)
        
        # Check value ranges
        if "total_points" in df.columns:
            points_range = (df["total_points"].min(), df["total_points"].max())
            if points_range[0] < -10:
                warnings.append(f"Unusually low points found: {points_range[0]}")
            if points_range[1] > 30:
                warnings.append(f"Unusually high points found: {points_range[1]}")
            stats["points_range"] = points_range
        
        if "minutes" in df.columns:
            max_minutes = df["minutes"].max()
            if max_minutes > 120:  # Allow extra time
                warnings.append(f"Minutes > 120 found: {max_minutes}")
        
        if "value" in df.columns:
            value_range = (df["value"].min(), df["value"].max())
            # Values are typically 35-150 (3.5m - 15.0m)
            if value_range[0] < 30 or value_range[1] > 200:
                warnings.append(f"Unusual value range: {value_range}")
            stats["value_range"] = value_range
        
        # Check for missing data
        null_counts = {}
        for col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                null_pct = null_count / len(df) * 100
                null_counts[col] = f"{null_pct:.1f}%"
                if null_pct > 50:
                    warnings.append(f"Column '{col}' has {null_pct:.1f}% missing values")
        
        if null_counts:
            stats["null_percentages"] = null_counts
        
        # Check gameweeks
        gw_col = "GW" if "GW" in df.columns else "round"
        if gw_col in df.columns:
            gws = sorted(df[gw_col].unique().to_list())
            stats["gameweeks"] = gws
            if len(gws) > 0 and (min(gws) < 1 or max(gws) > 38):
                warnings.append(f"Gameweek values out of range: {min(gws)} - {max(gws)}")
        
        # Determine validity
        is_valid = len(errors) == 0
        if self.strict and warnings:
            is_valid = False
            errors.extend(warnings)
            warnings = []
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )
    
    def validate_player_data(self, df: pl.DataFrame) -> ValidationResult:
        """
        Validate player data from FPL API.
        
        Args:
            df: DataFrame with player data
        
        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        stats = {}
        
        # Check required columns
        missing_cols = self.PLAYER_REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        if len(df) == 0:
            errors.append("DataFrame is empty")
            return ValidationResult(is_valid=False, errors=errors)
        
        stats["player_count"] = len(df)
        
        # Check unique IDs
        if "id" in df.columns:
            if df["id"].n_unique() != len(df):
                errors.append("Duplicate player IDs found")
        
        # Check teams
        if "team" in df.columns:
            team_count = df["team"].n_unique()
            stats["team_count"] = team_count
            if team_count != 20:
                warnings.append(f"Expected 20 teams, found {team_count}")
        
        # Check player count per position
        if "element_type" in df.columns:
            position_counts = df.group_by("element_type").agg(pl.count()).to_dicts()
            stats["position_counts"] = {p["element_type"]: p["count"] for p in position_counts}
        
        # Validate costs
        if "now_cost" in df.columns:
            cost_range = (df["now_cost"].min(), df["now_cost"].max())
            # Costs are in 0.1m units (e.g., 45 = 4.5m)
            if cost_range[0] < 35 or cost_range[1] > 160:
                warnings.append(f"Unusual cost range: {cost_range[0]/10}m - {cost_range[1]/10}m")
            stats["cost_range"] = (cost_range[0]/10, cost_range[1]/10)
        
        is_valid = len(errors) == 0
        if self.strict and warnings:
            is_valid = False
            errors.extend(warnings)
            warnings = []
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )
    
    def validate_fixtures(self, df: pl.DataFrame) -> ValidationResult:
        """
        Validate fixtures data.
        
        Args:
            df: DataFrame with fixtures data
        
        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        stats = {}
        
        missing_cols = self.FIXTURE_REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        if len(df) == 0:
            errors.append("DataFrame is empty")
            return ValidationResult(is_valid=False, errors=errors)
        
        stats["fixture_count"] = len(df)
        
        # Check gameweeks
        if "event" in df.columns:
            # Filter out null events (unscheduled fixtures)
            scheduled = df.filter(pl.col("event").is_not_null())
            if len(scheduled) > 0:
                gws = sorted(scheduled["event"].unique().to_list())
                stats["gameweeks_with_fixtures"] = len(gws)
        
        # Validate team counts
        if "team_h" in df.columns and "team_a" in df.columns:
            home_teams = df["team_h"].n_unique()
            away_teams = df["team_a"].n_unique()
            stats["unique_teams"] = max(home_teams, away_teams)
        
        is_valid = len(errors) == 0
        if self.strict and warnings:
            is_valid = False
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )
    
    def validate_features(
        self, 
        df: pl.DataFrame, 
        required_features: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate feature-engineered data before model training.
        
        Args:
            df: DataFrame with feature data
            required_features: List of feature columns that must be present
        
        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        stats = {}
        
        if len(df) == 0:
            errors.append("DataFrame is empty")
            return ValidationResult(is_valid=False, errors=errors)
        
        stats["row_count"] = len(df)
        stats["column_count"] = len(df.columns)
        
        # Check required features
        if required_features:
            missing = set(required_features) - set(df.columns)
            if missing:
                errors.append(f"Missing required features: {missing}")
        
        # Check for infinite values in numeric columns
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
        for col in numeric_cols:
            if df[col].is_infinite().any():
                errors.append(f"Infinite values found in column '{col}'")
        
        # Check null percentages
        high_null_cols = []
        for col in df.columns:
            null_pct = df[col].null_count() / len(df) * 100
            if null_pct > 20:
                high_null_cols.append((col, null_pct))
        
        if high_null_cols:
            for col, pct in high_null_cols[:5]:
                warnings.append(f"High null percentage in '{col}': {pct:.1f}%")
            stats["high_null_columns"] = len(high_null_cols)
        
        is_valid = len(errors) == 0
        if self.strict and warnings:
            is_valid = False
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )


def validate_and_report(
    df: pl.DataFrame, 
    data_type: str = "gameweek",
    strict: bool = False,
) -> bool:
    """
    Convenience function to validate data and log the report.
    
    Args:
        df: DataFrame to validate
        data_type: Type of data ('gameweek', 'player', 'fixture', 'features')
        strict: Whether to treat warnings as errors
    
    Returns:
        True if validation passed, False otherwise
    """
    validator = DataValidator(strict=strict)
    
    if data_type == "gameweek":
        result = validator.validate_gameweek_data(df)
    elif data_type == "player":
        result = validator.validate_player_data(df)
    elif data_type == "fixture":
        result = validator.validate_fixtures(df)
    elif data_type == "features":
        result = validator.validate_features(df)
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    logger.info(f"\n{result.summary()}")
    return result.is_valid
