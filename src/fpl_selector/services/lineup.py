"""
Lineup optimization service for FPL Selector.

Selects optimal starting XI and captain from user's 15-player squad.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import polars as pl

from fpl_selector.data.collectors import FPLAPICollector, CollectorConfig
from fpl_selector.data.storage import DataStorage
from fpl_selector.models.optimizer import TeamOptimizer, FPLConstraints
from fpl_selector.services.predictions import PredictionService

logger = logging.getLogger(__name__)


@dataclass
class LineupRecommendation:
    """Recommended lineup for a gameweek."""
    starting_xi: List[Dict]
    bench: List[Dict]  # Ordered by priority
    captain: Dict
    vice_captain: Dict
    formation: str
    expected_points: float
    
    def display(self) -> str:
        """Format lineup for display."""
        lines = [
            f"Formation: {self.formation}",
            f"Expected Points: {self.expected_points:.1f}",
            "",
            f"Captain: {self.captain.get('web_name', 'Unknown')} ({self.captain.get('predicted_points', 0):.1f})",
            f"Vice Captain: {self.vice_captain.get('web_name', 'Unknown')}",
            "",
            "Starting XI:",
        ]
        
        # Group by position
        by_pos = {"GK": [], "DEF": [], "MID": [], "FWD": []}
        for p in self.starting_xi:
            pos = p.get("position", "MID")
            by_pos[pos].append(p)
        
        for pos in ["GK", "DEF", "MID", "FWD"]:
            players = by_pos[pos]
            if players:
                names = [f"{p.get('web_name', '?')} ({p.get('predicted_points', 0):.1f})" 
                        for p in players]
                lines.append(f"  {pos}: {', '.join(names)}")
        
        lines.append("")
        lines.append("Bench:")
        for i, p in enumerate(self.bench, 1):
            lines.append(f"  {i}. {p.get('web_name', '?')} ({p.get('predicted_points', 0):.1f})")
        
        return "\n".join(lines)


class LineupService:
    """
    Service for optimizing starting lineup from a 15-player squad.
    
    Considers:
    - Predicted points for each player
    - Formation constraints (min 3 DEF, min 2 MID, min 1 FWD)
    - Captain selection for 2x points
    """
    
    def __init__(
        self,
        storage: Optional[DataStorage] = None,
        prediction_service: Optional[PredictionService] = None,
    ):
        """
        Initialize lineup service.
        
        Args:
            storage: Data storage instance
            prediction_service: Prediction service instance
        """
        self.storage = storage or DataStorage()
        self.predictions = prediction_service or PredictionService(self.storage)
        self.optimizer = TeamOptimizer()
        self.api_collector = FPLAPICollector(CollectorConfig())
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def fetch_user_squad(self, team_id: int) -> Tuple[Dict, pl.DataFrame]:
        """
        Fetch user's squad with prediction data.
        
        Args:
            team_id: FPL team ID
        
        Returns:
            Tuple of (team info, squad DataFrame with predictions)
        """
        # Get user's picks
        team_info = await self.api_collector.get_user_team(team_id)
        current_gw = await self.api_collector.get_current_gameweek()
        picks_data = await self.api_collector.get_user_picks(team_id, current_gw)
        
        picks = picks_data.get("picks", [])
        if not picks:
            raise ValueError(f"Could not fetch picks for team {team_id}")
        
        player_ids = [p["element"] for p in picks]
        
        # Get predictions for all players
        all_predictions = self.predictions.get_predictions()
        
        # Filter to user's squad
        squad = all_predictions.filter(pl.col("id").is_in(player_ids))
        
        # Add pick metadata (is_captain, multiplier, position in squad)
        picks_df = pl.DataFrame(picks)
        squad = squad.join(
            picks_df.select(["element", "is_captain", "is_vice_captain", "multiplier"]),
            left_on="id",
            right_on="element",
            how="left",
        )
        
        return team_info, squad
    
    def optimize_lineup(self, team_id: int) -> LineupRecommendation:
        """
        Generate optimal lineup for a team.
        
        Args:
            team_id: FPL team ID
        
        Returns:
            LineupRecommendation with optimal starting XI
        """
        # Fetch squad
        team_info, squad = asyncio.run(self.fetch_user_squad(team_id))
        
        if len(squad) == 0:
            raise ValueError(f"No squad data for team {team_id}")
        
        # Select starting XI
        starting_xi, bench = self._select_starting_xi(squad)
        
        # Select captain
        captain, vice_captain = self._select_captain(starting_xi)
        
        # Calculate expected points
        expected = sum(p.get("predicted_points", 0) for p in starting_xi)
        expected += captain.get("predicted_points", 0)  # Captain gets double
        
        # Determine formation
        formation = self._get_formation(starting_xi)
        
        return LineupRecommendation(
            starting_xi=starting_xi,
            bench=bench,
            captain=captain,
            vice_captain=vice_captain,
            formation=formation,
            expected_points=expected,
        )
    
    def _select_starting_xi(
        self,
        squad: pl.DataFrame,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Select best 11 from 15-player squad.
        
        Constraints:
        - Exactly 1 GK
        - At least 3 DEF
        - At least 2 MID
        - At least 1 FWD
        - Maximum 11 players
        """
        # Sort by predicted points within each position
        squad_sorted = squad.sort("predicted_points", descending=True)
        
        # Group by position
        by_position: Dict[str, List[Dict]] = {"GK": [], "DEF": [], "MID": [], "FWD": []}
        for row in squad_sorted.iter_rows(named=True):
            pos = row.get("position", "MID")
            if pos in by_position:
                by_position[pos].append(row)
        
        starting_xi = []
        bench = []
        
        # Select required players first
        # 1 GK (must)
        if by_position["GK"]:
            starting_xi.append(by_position["GK"][0])
            bench.extend(by_position["GK"][1:])
        
        # Track how many we've selected per position
        selected = {"DEF": 0, "MID": 0, "FWD": 0}
        minimums = {"DEF": 3, "MID": 2, "FWD": 1}
        
        # First pass: select minimums
        for pos, min_count in minimums.items():
            for player in by_position[pos][:min_count]:
                starting_xi.append(player)
                selected[pos] += 1
        
        # Second pass: fill remaining spots with best players
        remaining_slots = 11 - len(starting_xi)
        
        # Pool remaining outfield players
        remaining = []
        for pos in ["DEF", "MID", "FWD"]:
            remaining.extend(by_position[pos][selected[pos]:])
        
        # Sort by predicted points
        remaining.sort(key=lambda x: x.get("predicted_points", 0), reverse=True)
        
        # Add best remaining to starting XI
        for player in remaining[:remaining_slots]:
            starting_xi.append(player)
        
        # Rest go to bench
        bench.extend(remaining[remaining_slots:])
        
        return starting_xi, bench
    
    def _select_captain(
        self,
        starting_xi: List[Dict],
    ) -> Tuple[Dict, Dict]:
        """Select captain and vice captain based on predictions."""
        # Sort by predicted points
        sorted_xi = sorted(
            starting_xi,
            key=lambda x: x.get("predicted_points", 0),
            reverse=True
        )
        
        captain = sorted_xi[0] if sorted_xi else {}
        vice_captain = sorted_xi[1] if len(sorted_xi) > 1 else {}
        
        return captain, vice_captain
    
    def _get_formation(self, starting_xi: List[Dict]) -> str:
        """Determine formation string from starting XI."""
        counts = {"DEF": 0, "MID": 0, "FWD": 0}
        
        for player in starting_xi:
            pos = player.get("position", "")
            if pos in counts:
                counts[pos] += 1
        
        return f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"
    
    def compare_with_current(
        self,
        team_id: int,
    ) -> Dict:
        """
        Compare optimal lineup with user's current selection.
        
        Returns differences and potential points gain.
        """
        # Get current picks
        team_info, squad = asyncio.run(self.fetch_user_squad(team_id))
        
        # Get optimal
        optimal = self.optimize_lineup(team_id)
        
        # Get current starting XI (those with multiplier > 0 in picks)
        current_starters = squad.filter(pl.col("multiplier") > 0).to_dicts()
        current_bench = squad.filter(pl.col("multiplier") == 0).to_dicts()
        
        # Current expected points
        current_captain = squad.filter(pl.col("is_captain") == True).to_dicts()
        current_expected = sum(p.get("predicted_points", 0) for p in current_starters)
        if current_captain:
            current_expected += current_captain[0].get("predicted_points", 0)
        
        # Calculate difference
        points_diff = optimal.expected_points - current_expected
        
        # Find changed players
        optimal_ids = {p.get("id") for p in optimal.starting_xi}
        current_ids = {p.get("id") for p in current_starters}
        
        return {
            "current_expected": current_expected,
            "optimal_expected": optimal.expected_points,
            "points_difference": points_diff,
            "players_to_bench": [p for p in current_starters if p.get("id") not in optimal_ids],
            "players_to_start": [p for p in optimal.starting_xi if p.get("id") not in current_ids],
            "captain_change": optimal.captain.get("id") != (current_captain[0].get("id") if current_captain else None),
        }
