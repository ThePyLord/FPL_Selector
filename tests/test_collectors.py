"""Tests for data collectors."""

import pytest
import polars as pl

from fpl_selector.data.collectors import (
    VastaavCollector,
    FPLAPICollector,
    CollectorConfig,
)


@pytest.fixture
def config():
    """Get default collector config."""
    return CollectorConfig(timeout_seconds=60, max_retries=2)


class TestVastaavCollector:
    """Tests for VastaavCollector."""
    
    def test_current_season_calculation(self, config):
        """Test that current season is calculated correctly."""
        collector = VastaavCollector(config)
        season = collector.current_season
        
        # Should be in format YYYY-YY
        assert len(season) == 7
        assert "-" in season
        
        year1, year2 = season.split("-")
        assert len(year1) == 4
        assert len(year2) == 2
        assert int(year2) == (int(year1) + 1) % 100
    
    @pytest.mark.asyncio
    async def test_collect_merged_gameweeks(self, config):
        """Test collecting merged gameweek data."""
        collector = VastaavCollector(config)
        
        # Use a known past season for reliability
        df = await collector.collect(season="2023-24")
        
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
        assert "name" in df.columns or "element" in df.columns
        assert "total_points" in df.columns
    
    @pytest.mark.asyncio
    async def test_collect_specific_gameweeks(self, config):
        """Test collecting specific gameweeks."""
        collector = VastaavCollector(config)
        
        df = await collector.collect(season="2023-24", gameweeks=[1, 2, 3])
        
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
        assert "gw" in df.columns or "GW" in df.columns


class TestFPLAPICollector:
    """Tests for FPLAPICollector."""
    
    @pytest.mark.asyncio
    async def test_get_bootstrap_data(self, config):
        """Test fetching bootstrap data."""
        collector = FPLAPICollector(config)
        
        data = await collector.get_bootstrap_data()
        
        assert isinstance(data, dict)
        assert "elements" in data
        assert "teams" in data
        assert "events" in data
        assert len(data["elements"]) > 500  # Should have 600+ players
        assert len(data["teams"]) == 20
    
    @pytest.mark.asyncio
    async def test_collect_players(self, config):
        """Test collecting player data."""
        collector = FPLAPICollector(config)
        
        df = await collector.collect()
        
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 500
        assert "id" in df.columns
        assert "web_name" in df.columns or "second_name" in df.columns
    
    @pytest.mark.asyncio
    async def test_get_fixtures(self, config):
        """Test fetching fixtures."""
        collector = FPLAPICollector(config)
        
        df = await collector.get_fixtures()
        
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
        assert "team_h" in df.columns
        assert "team_a" in df.columns
    
    @pytest.mark.asyncio
    async def test_get_current_gameweek(self, config):
        """Test getting current gameweek."""
        collector = FPLAPICollector(config)
        
        gw = await collector.get_current_gameweek()
        
        assert isinstance(gw, int)
        assert 1 <= gw <= 38
