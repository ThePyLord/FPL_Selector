"""
Transfer recommendation service for FPL Selector.

Analyzes user's squad and suggests optimal transfers.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import polars as pl

from fpl_selector.data.collectors import FPLAPICollector, CollectorConfig
from fpl_selector.data.storage import DataStorage
from fpl_selector.models.optimizer import TeamOptimizer
from fpl_selector.services.predictions import PredictionService

logger = logging.getLogger(__name__)


@dataclass
class TransferSuggestion:
    """A suggested transfer."""
    player_out: Dict
    player_in: Dict
    points_gain: float
    cost_change: float  # Positive = money saved
    
    def __str__(self) -> str:
        out_name = self.player_out.get("web_name", "Unknown")
        in_name = self.player_in.get("web_name", "Unknown")
        return f"{out_name} → {in_name} (+{self.points_gain:.1f} pts)"


class TransferService:
    """
    Service for analyzing and recommending transfers.
    
    Fetches user's current squad and suggests improvements
    based on predicted points and value.
    """
    
    def __init__(
        self,
        storage: Optional[DataStorage] = None,
        prediction_service: Optional[PredictionService] = None,
    ):
        """
        Initialize transfer service.
        
        Args:
            storage: Data storage instance
            prediction_service: Prediction service instance
        """
        self.storage = storage or DataStorage()
        self.predictions = prediction_service or PredictionService(self.storage)
        self.optimizer = TeamOptimizer()
        self.api_collector = FPLAPICollector(CollectorConfig())
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def get_user_squad(self, team_id: int) -> Tuple[Dict, List[Dict]]:
        """
        Fetch user's current squad.
        
        Args:
            team_id: FPL team ID
        
        Returns:
            Tuple of (team_info, list of player picks)
        """
        team_info = await self.api_collector.get_user_team(team_id)
        current_gw = await self.api_collector.get_current_gameweek()
        picks_data = await self.api_collector.get_user_picks(team_id, current_gw)
        
        return team_info, picks_data.get("picks", [])
    
    def analyze_squad(
        self,
        team_id: int,
        free_transfers: int = 1,
        budget_remaining: float = 0.0,
    ) -> Dict:
        """
        Analyze squad and generate transfer suggestions.
        
        Args:
            team_id: FPL team ID
            free_transfers: Available free transfers
            budget_remaining: Money in bank (£m)
        
        Returns:
            Analysis results including suggestions
        """
        # Fetch squad
        team_info, picks = asyncio.run(self.get_user_squad(team_id))
        
        if not picks:
            return {"error": "Could not fetch squad picks"}
        
        # Get player IDs
        player_ids = [p["element"] for p in picks]
        
        # Get all players with predictions
        all_players = self.predictions.get_predictions()
        
        # Get squad details
        squad = all_players.filter(pl.col("id").is_in(player_ids))
        
        # Analyze squad composition
        analysis = self._analyze_composition(squad)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(
            squad=squad,
            all_players=all_players,
            free_transfers=free_transfers,
            budget_remaining=budget_remaining,
        )
        
        # Calculate squad metrics
        total_value = squad.select(pl.col("now_cost").sum()).item() / 10
        avg_form = squad.select(pl.col("form").cast(pl.Float64).mean()).item()
        
        entry = team_info.get("entry", team_info)
        
        return {
            "team_name": entry.get("name", "Unknown"),
            "overall_rank": entry.get("summary_overall_rank"),
            "total_points": entry.get("summary_overall_points"),
            "squad_value": total_value,
            "average_form": avg_form,
            "composition": analysis,
            "suggestions": suggestions,
            "free_transfers": free_transfers,
            "budget": budget_remaining,
        }
    
    def _analyze_composition(self, squad: pl.DataFrame) -> Dict:
        """Analyze squad composition."""
        if "position" not in squad.columns:
            return {}
        
        by_position = squad.group_by("position").agg(pl.count())
        composition = {r["position"]: r["count"] for r in by_position.iter_rows(named=True)}
        
        # Check for issues
        issues = []
        
        # Check team diversity
        if "team" in squad.columns:
            team_counts = squad.group_by("team").agg(pl.count())
            max_from_team = team_counts.select(pl.col("count").max()).item()
            if max_from_team > 3:
                issues.append(f"Too many players from one team ({max_from_team})")
        
        return {
            "positions": composition,
            "issues": issues,
        }
    
    def _generate_suggestions(
        self,
        squad: pl.DataFrame,
        all_players: pl.DataFrame,
        free_transfers: int,
        budget_remaining: float,
    ) -> List[TransferSuggestion]:
        """Generate transfer suggestions."""
        suggestions = []
        
        # Get squad player IDs
        squad_ids = squad.select("id").to_series().to_list()
        
        # Find lowest predicted points in squad (excluding GK for now)
        squad_sorted = squad.filter(
            pl.col("position") != "GK"
        ).sort("predicted_points")
        
        for i, row in enumerate(squad_sorted.head(free_transfers + 2).iter_rows(named=True)):
            position = row.get("position")
            current_price = row.get("now_cost", 0)
            current_pred = row.get("predicted_points", 0)
            max_price = current_price + (budget_remaining * 10)
            
            # Find better alternatives
            alternatives = all_players.filter(
                (pl.col("position") == position) &
                (pl.col("now_cost") <= max_price) &
                (~pl.col("id").is_in(squad_ids)) &
                (pl.col("predicted_points") > current_pred)
            ).sort("predicted_points", descending=True)
            
            if len(alternatives) > 0:
                best = alternatives.head(1).to_dicts()[0]
                gain = best.get("predicted_points", 0) - current_pred
                cost_change = (current_price - best.get("now_cost", 0)) / 10
                
                suggestions.append(TransferSuggestion(
                    player_out=row,
                    player_in=best,
                    points_gain=gain,
                    cost_change=cost_change,
                ))
        
        # Sort by points gain
        suggestions.sort(key=lambda x: x.points_gain, reverse=True)
        
        return suggestions[:5]  # Top 5 suggestions
