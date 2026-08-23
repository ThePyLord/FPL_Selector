"""
FPL Selector CLI.

Command-line interface for the FPL Selector tool.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from fpl_selector.data.collectors import VastaavCollector, FPLAPICollector, CollectorConfig
from fpl_selector.data.storage import DataStorage
from fpl_selector.data.validators import DataValidator

# Initialize Typer app
app = typer.Typer(
    name="fpl",
    help="FPL Selector - AI-powered Fantasy Premier League assistant",
    add_completion=False,
)

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def get_storage() -> DataStorage:
    """Get the data storage instance."""
    return DataStorage()


@app.command()
def update(
    season: Optional[str] = typer.Option(
        None, "--season", "-s", help="Season to update (e.g., '2025-26')"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force update even if data is fresh"
    ),
):
    """
    Update all data from FPL API and Vaastav dataset.
    
    This fetches the latest player data, fixtures, and gameweek history.
    """
    storage = get_storage()
    config = CollectorConfig()
    validator = DataValidator()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Check data freshness
        freshness = storage.get_data_freshness()
        if not force and freshness.get("players"):
            hours_old = (datetime.now() - freshness["players"]).total_seconds() / 3600
            if hours_old < 1:
                console.print(f"[yellow]Data was updated {hours_old:.1f} hours ago. Use --force to update anyway.[/yellow]")
        
        # Collect from FPL API
        task = progress.add_task("Fetching player data from FPL API...", total=None)
        try:
            fpl_collector = FPLAPICollector(config)
            
            # Get bootstrap data
            bootstrap = asyncio.run(fpl_collector.get_bootstrap_data())
            players_df = fpl_collector._parse_players(bootstrap)
            
            # Validate
            result = validator.validate_player_data(players_df)
            if not result.is_valid:
                console.print(f"[red]Player data validation failed:[/red]")
                for error in result.errors:
                    console.print(f"  [red]• {error}[/red]")
                raise typer.Exit(1)
            
            storage.save_players(players_df)
            progress.update(task, description=f"[green]✓[/green] Fetched {len(players_df)} players")
            
            # Get fixtures
            progress.update(task, description="Fetching fixtures...")
            fixtures_df = asyncio.run(fpl_collector.get_fixtures())
            result = validator.validate_fixtures(fixtures_df)
            if result.is_valid:
                storage.save_fixtures(fixtures_df)
                progress.update(task, description=f"[green]✓[/green] Fetched {len(fixtures_df)} fixtures")
            else:
                console.print(f"[yellow]Fixture validation warnings: {result.warnings}[/yellow]")
            
            # Get current gameweek
            current_gw = asyncio.run(fpl_collector.get_current_gameweek())
            
        except Exception as e:
            console.print(f"[red]Failed to fetch FPL API data: {e}[/red]")
            raise typer.Exit(1)
        
        # Collect from Vaastav
        progress.update(task, description="Fetching gameweek history from Vaastav...")
        try:
            vaastav_collector = VastaavCollector(config)
            target_season = season or vaastav_collector.current_season
            
            gw_df = asyncio.run(vaastav_collector.collect(season=target_season))
            
            result = validator.validate_gameweek_data(gw_df)
            if not result.is_valid:
                console.print(f"[red]Gameweek data validation failed:[/red]")
                for error in result.errors:
                    console.print(f"  [red]• {error}[/red]")
                raise typer.Exit(1)
            
            storage.save_gameweeks(gw_df, target_season)
            progress.update(task, description=f"[green]✓[/green] Fetched {len(gw_df)} gameweek records")
            
            # Also fetch teams
            teams_df = asyncio.run(vaastav_collector.collect_teams(target_season))
            storage.save_teams(teams_df, target_season)
            
        except Exception as e:
            console.print(f"[red]Failed to fetch Vaastav data: {e}[/red]")
            raise typer.Exit(1)
    
    # Summary
    console.print()
    console.print("[bold green]✓ Data update complete![/bold green]")
    console.print(f"  Season: {target_season}")
    console.print(f"  Current gameweek: {current_gw}")
    console.print(f"  Players: {len(players_df)}")
    console.print(f"  Fixtures: {len(fixtures_df)}")
    console.print(f"  Gameweek records: {len(gw_df)}")


@app.command()
def predict(
    position: Optional[str] = typer.Option(
        None, "--position", "-p", help="Filter by position (GK, DEF, MID, FWD)"
    ),
    top: int = typer.Option(
        20, "--top", "-n", help="Number of players to show"
    ),
    min_minutes: int = typer.Option(
        90, "--min-minutes", "-m", help="Minimum minutes played (recent form filter)"
    ),
):
    """
    Show predicted points for the next gameweek.
    
    Predictions are based on form, fixtures, and historical performance.
    """
    storage = get_storage()
    
    # Try to use ML model predictions
    try:
        from fpl_selector.models.predictor import PointsPredictor
        from fpl_selector.features.engineering import FeatureEngineer
        
        # Check if we have trained models
        models = storage.list_models()
        if models:
            # Load features and generate predictions
            features_df = storage.load_features()
            # features_df = features_df
            predictor = PointsPredictor(model_dir=storage.base_path / "models")
            predictor.load()
            
            predictions = predictor.predict(features_df)
            
            # Get latest gameweek per player
            gw_col = "GW" if "GW" in predictions.columns else "round"
            latest_gw = predictions[gw_col].max()
            df = predictions.filter(pl.col(gw_col) == latest_gw)
            
            console.print()
            console.print(f"[bold]Next Gameweek Predictions (GW {latest_gw + 1})[/bold]")
            console.print("[dim](XGBoost model trained on 2025-26 data)[/dim]")
            
            use_ml = True
        else:
            raise FileNotFoundError("No trained models")
            
    except (FileNotFoundError, Exception) as e:
        # Fallback to MVP mode
        try:
            df = storage.load_players()
        except FileNotFoundError:
            console.print("[red]No player data found. Run 'fpl update' first.[/red]")
            raise typer.Exit(1)
        
        console.print()
        console.print("[bold]Next Gameweek Predictions[/bold]")
        console.print("[dim]NOT USING ANY MODELS. TRAIN ON DATA TO USE ML PREDICTION [/dim]")
        console.print("[dim](MVP mode - using form as proxy for predictions)[/dim]")
        use_ml = False
    
    console.print()
    
    # Filter by position
    if position:
        position = position.upper()
        if position not in ["GK", "DEF", "MID", "FWD"]:
            console.print(f"[red]Invalid position: {position}. Use GK, DEF, MID, or FWD.[/red]")
            raise typer.Exit(1)
        df = df.filter(pl.col("position") == position)
    
    # Filter by minutes
    if "minutes" in df.columns:
        df = df.filter(pl.col("minutes") >= min_minutes)
    
    # Sort by predictions or form
    if use_ml and "predicted_points" in df.columns:
        df = df.sort("predicted_points", descending=True)
    else:
        sort_col = "form" if "form" in df.columns else "total_points"
        if sort_col in df.columns:
            df = df.sort(sort_col, descending=True)
    
    df = df.head(top)
    
    # Build display table
    table = Table(title=f"Top {top} Players by Predicted Points")
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Player", style="cyan")
    table.add_column("Team", style="green")
    table.add_column("Pos", style="yellow")
    table.add_column("Price", justify="right")
    if use_ml:
        table.add_column("Pred", justify="right", style="bold magenta")
    else:
        table.add_column("Form", justify="right", style="bold")
    table.add_column("Pts", justify="right")
    table.add_column("Selected", justify="right", style="dim")
    
    for i, row in enumerate(df.iter_rows(named=True), 1):
        price = row.get("now_cost", row.get("value", 0))
        if price > 100:  # API returns in 0.1m units
            price = price / 10
        
        pts = row.get("total_points", 0)
        selected = row.get("selected_by_percent", row.get("selected", "-"))
        
        if use_ml:
            pred = row.get("predicted_points", 0)
            pred_str = f"{pred:.1f}"
        else:
            pred_str = str(row.get("form", "-"))
        
        table.add_row(
            str(i),
            row.get("web_name", row.get("name", "Unknown")),
            row.get("team_name", row.get("team", "")),
            row.get("position", "?"),
            f"£{price:.1f}m",
            pred_str,
            str(pts),
            f"{selected}%" if isinstance(selected, (int, float)) else str(selected),
        )
    
    console.print(table)


@app.command()
def transfers(
    team_id: int = typer.Argument(..., help="Your FPL team ID"),
    free_transfers: int = typer.Option(1, "--ft", help="Number of free transfers"),
    budget: Optional[float] = typer.Option(None, "--budget", "-b", help="Budget in bank (£m)"),
):
    """
    Get transfer recommendations for your team.
    
    Analyzes your current squad and suggests beneficial transfers.
    """
    console.print()
    console.print(f"[bold]Transfer Analysis for Team {team_id}[/bold]")
    console.print()
    
    config = CollectorConfig()
    fpl_collector = FPLAPICollector(config)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching your team...", total=None)
        
        try:
            team_info = asyncio.run(fpl_collector.get_user_team(team_id))
            current_gw = asyncio.run(fpl_collector.get_current_gameweek())
            picks = asyncio.run(fpl_collector.get_user_picks(team_id, current_gw))
        except Exception as e:
            console.print(f"[red]Failed to fetch team data: {e}[/red]")
            console.print("[dim]Make sure your team ID is correct and your team is public.[/dim]")
            raise typer.Exit(1)
        
        progress.update(task, description="[green]✓[/green] Team data loaded")
    
    # Display team info
    entry = team_info.get("entry", team_info)
    team_name = entry.get("name", "Unknown")
    overall_rank = entry.get("summary_overall_rank", "N/A")
    
    console.print(f"Team: [cyan]{team_name}[/cyan]")
    console.print(f"Overall Rank: {overall_rank:,}" if isinstance(overall_rank, int) else f"Overall Rank: {overall_rank}")
    console.print(f"Free Transfers: {free_transfers}")
    console.print()
    
    # TODO: Implement actual transfer optimization in Week 3
    console.print("[yellow]Transfer recommendations coming in Week 3 of development.[/yellow]")
    console.print("[dim]For now, use 'fpl predict' to see top players by form.[/dim]")


@app.command()
def lineup(
    team_id: int = typer.Argument(..., help="Your FPL team ID"),
):
    """
    Get optimal starting XI and captain picks.
    
    Analyzes your 15-player squad and suggests the best lineup.
    """
    console.print()
    console.print(f"[bold]Lineup Optimization for Team {team_id}[/bold]")
    console.print()
    
    # TODO: Implement lineup optimization in Week 3
    console.print("[yellow]Lineup optimization coming in Week 3 of development.[/yellow]")
    console.print("[dim]For now, use 'fpl predict' to see top players by form.[/dim]")


@app.command()
def status():
    """
    Show the current status of local data and models.
    """
    storage = get_storage()
    
    console.print()
    console.print("[bold]FPL Selector Status[/bold]")
    console.print()
    
    # Data freshness
    freshness = storage.get_data_freshness()
    
    table = Table(title="Data Freshness")
    table.add_column("Data Type", style="cyan")
    table.add_column("Last Updated", style="green")
    table.add_column("Status")
    
    for data_type, timestamp in freshness.items():
        if timestamp:
            age_hours = (datetime.now() - timestamp).total_seconds() / 3600
            if age_hours < 24:
                status = "[green]Fresh[/green]"
            elif age_hours < 168:  # 1 week
                status = "[yellow]Stale[/yellow]"
            else:
                status = "[red]Outdated[/red]"
            time_str = timestamp.strftime("%Y-%m-%d %H:%M")
        else:
            time_str = "Never"
            status = "[red]Missing[/red]"
        
        table.add_row(data_type.title(), time_str, status)
    
    console.print(table)
    console.print()
    
    # Available seasons
    try:
        seasons = storage.list_available_seasons()
        console.print(f"Available seasons: {', '.join(seasons) if seasons else 'None'}")
    except Exception:
        console.print("Available seasons: None")
    
    # Models
    try:
        models = storage.list_models()
        console.print(f"Trained models: {', '.join(models) if models else 'None'}")
    except Exception:
        console.print("Trained models: None")


@app.command()
def version():
    """Show version information."""
    from fpl_selector import __version__
    console.print(f"FPL Selector v{__version__}")


# Import polars for predict command
import polars as pl


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
