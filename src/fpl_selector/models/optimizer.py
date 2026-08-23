"""
Team optimization for FPL Selector.

This module solves the constrained optimization problem of selecting
the best FPL squad given budget, position, and team constraints.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class FPLConstraints:
    """Constraints for FPL team selection."""
    budget: float = 100.0  # Total budget in millions
    squad_size: int = 15
    max_per_team: int = 3
    
    # Position requirements
    goalkeepers: int = 2
    defenders: int = 5
    midfielders: int = 5
    forwards: int = 3
    
    # Starting XI constraints
    min_defenders: int = 3
    min_midfielders: int = 2
    min_forwards: int = 1


@dataclass
class TeamSelection:
    """Result of team optimization."""
    squad: List[Dict]  # 15 players
    starting_xi: List[Dict]  # 11 players
    bench: List[Dict]  # 4 players (ordered)
    captain: Dict
    vice_captain: Dict
    total_cost: float
    expected_points: float
    formation: str  # e.g., "3-4-3"


class TeamOptimizer:
    """
    Optimizer for FPL team selection.
    
    Uses linear programming or greedy heuristics to find optimal
    squad composition given constraints.
    """
    
    def __init__(self, constraints: Optional[FPLConstraints] = None):
        """
        Initialize optimizer.
        
        Args:
            constraints: FPL constraints to apply
        """
        self.constraints = constraints or FPLConstraints()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def optimize_squad(
        self,
        players_df: pl.DataFrame,
        predictions_col: str = "predicted_points",
        price_col: str = "now_cost",
    ) -> TeamSelection:
        """
        Select optimal 15-player squad.
        
        Args:
            players_df: DataFrame with player data and predictions
            predictions_col: Column name for predicted points
            price_col: Column name for player price
        
        Returns:
            TeamSelection with optimal squad
        """
        # For MVP, use greedy selection
        # TODO: Implement proper LP optimization in later iteration
        
        squad = self._greedy_selection(
            players_df,
            predictions_col,
            price_col,
        )
        
        # Select starting XI
        starting_xi, bench = self._select_starting_xi(squad, predictions_col)
        
        # Select captain
        captain, vice_captain = self._select_captains(starting_xi, predictions_col)
        
        # Calculate totals
        total_cost = sum(p.get(price_col, 0) for p in squad) / 10
        expected_points = sum(p.get(predictions_col, 0) for p in starting_xi)
        
        # Determine formation
        formation = self._get_formation(starting_xi)
        
        return TeamSelection(
            squad=squad,
            starting_xi=starting_xi,
            bench=bench,
            captain=captain,
            vice_captain=vice_captain,
            total_cost=total_cost,
            expected_points=expected_points,
            formation=formation,
        )
    
    def suggest_transfers(
        self,
        current_squad: List[int],  # Player IDs
        players_df: pl.DataFrame,
        predictions_col: str = "predicted_points",
        free_transfers: int = 1,
        budget_remaining: float = 0.0,
    ) -> List[Tuple[Dict, Dict, float]]:
        """
        Suggest transfers to improve squad.
        
        Args:
            current_squad: List of current player IDs
            players_df: All available players
            predictions_col: Prediction column
            free_transfers: Available free transfers
            budget_remaining: Money in bank
        
        Returns:
            List of (player_out, player_in, points_gain) tuples
        """
        # Filter to current squad
        squad_df = players_df.filter(pl.col("id").is_in(current_squad))
        
        # Find worst performers in squad
        squad_df = squad_df.sort(predictions_col)
        
        suggestions = []
        
        for row in squad_df.head(5).iter_rows(named=True):
            # Find better alternatives at same position
            position = row.get("position", row.get("element_type"))
            current_price = row.get("now_cost", 0)
            max_price = current_price + (budget_remaining * 10)
            
            alternatives = players_df.filter(
                (pl.col("position") == position) &
                (pl.col("now_cost") <= max_price) &
                (~pl.col("id").is_in(current_squad))
            ).sort(predictions_col, descending=True)
            
            if len(alternatives) > 0:
                best_alt = alternatives.head(1).to_dicts()[0]
                gain = best_alt.get(predictions_col, 0) - row.get(predictions_col, 0)
                
                if gain > 0:
                    suggestions.append((row, best_alt, gain))
        
        # Sort by points gain
        suggestions.sort(key=lambda x: x[2], reverse=True)
        
        return suggestions[:free_transfers + 3]  # Show a few extra options
    
    def _greedy_selection(
        self,
        players_df: pl.DataFrame,
        predictions_col: str,
        price_col: str,
    ) -> List[Dict]:
        """Greedy algorithm for squad selection."""
        c = self.constraints
        budget = c.budget * 10  # Convert to 0.1m units
        
        squad = []
        team_counts: Dict[int, int] = {}
        
        position_targets = {
            "GK": c.goalkeepers,
            "DEF": c.defenders,
            "MID": c.midfielders,
            "FWD": c.forwards,
        }
        
        # Sort by value (predicted points per price)
        players = players_df.with_columns(
            (pl.col(predictions_col) / (pl.col(price_col) / 10)).alias("value")
        ).sort("value", descending=True)
        
        for pos, target in position_targets.items():
            pos_players = players.filter(pl.col("position") == pos)
            selected = 0
            
            for row in pos_players.iter_rows(named=True):
                if selected >= target:
                    break
                
                player_id = row.get("id")
                team_id = row.get("team")
                price = row.get(price_col, 0)
                
                # Check constraints
                if team_counts.get(team_id, 0) >= c.max_per_team:
                    continue
                if price > budget:
                    continue
                
                # Select player
                squad.append(row)
                budget -= price
                team_counts[team_id] = team_counts.get(team_id, 0) + 1
                selected += 1
        
        return squad
    
    def _select_starting_xi(
        self,
        squad: List[Dict],
        predictions_col: str,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Select best starting 11 from squad."""
        c = self.constraints
        
        # Group by position
        by_position = {"GK": [], "DEF": [], "MID": [], "FWD": []}
        for player in squad:
            pos = player.get("position", "MID")
            by_position[pos].append(player)
        
        # Sort each position by predicted points
        for pos in by_position:
            by_position[pos].sort(
                key=lambda x: x.get(predictions_col, 0),
                reverse=True
            )
        
        starting_xi = []
        bench = []
        
        # Must have 1 GK
        if by_position["GK"]:
            starting_xi.append(by_position["GK"][0])
            bench.extend(by_position["GK"][1:])
        
        # Get minimum required for each outfield position
        for pos, min_count in [("DEF", c.min_defenders), 
                                ("MID", c.min_midfielders),
                                ("FWD", c.min_forwards)]:
            starting_xi.extend(by_position[pos][:min_count])
            remaining = by_position[pos][min_count:]
            
            # The rest go to a pool
            by_position[f"{pos}_remaining"] = remaining
        
        # Fill remaining spots (11 - 1 GK - minimums = flexible spots)
        spots_filled = 1 + c.min_defenders + c.min_midfielders + c.min_forwards
        flex_spots = 11 - spots_filled
        
        # Pool remaining players
        flex_pool = []
        for pos in ["DEF", "MID", "FWD"]:
            flex_pool.extend(by_position.get(f"{pos}_remaining", []))
        
        flex_pool.sort(key=lambda x: x.get(predictions_col, 0), reverse=True)
        starting_xi.extend(flex_pool[:flex_spots])
        bench.extend(flex_pool[flex_spots:])
        
        return starting_xi, bench
    
    def _select_captains(
        self,
        starting_xi: List[Dict],
        predictions_col: str,
    ) -> Tuple[Dict, Dict]:
        """Select captain and vice captain."""
        sorted_xi = sorted(
            starting_xi,
            key=lambda x: x.get(predictions_col, 0),
            reverse=True
        )
        
        captain = sorted_xi[0] if sorted_xi else {}
        vice_captain = sorted_xi[1] if len(sorted_xi) > 1 else {}
        
        return captain, vice_captain
    
    def _get_formation(self, starting_xi: List[Dict]) -> str:
        """Determine formation from starting XI."""
        counts = {"DEF": 0, "MID": 0, "FWD": 0}
        
        for player in starting_xi:
            pos = player.get("position", "")
            if pos in counts:
                counts[pos] += 1
        
        return f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"
