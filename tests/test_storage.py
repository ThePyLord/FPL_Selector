import pytest
# import polars as pl

from fpl_selector.data.storage import DataStorage


class TestDataStorage:

	def test_correct_paths(self):
		ds = DataStorage()