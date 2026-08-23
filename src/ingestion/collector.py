import asyncio
import datetime
import io
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import polars as pl
# import pandas as pd
from aiohttp import ClientSession

class VastaavCollector(PipelineStep):

	def __init__(self, config):
		super().__init__("vastaav_collector", config)
		self.logger = logging.getLogger(__name__)
		self.base_url = config.get("sources", {}).get(
			"vastaav_base_url",
			"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data",
		)
		curr_date = datetime.datetime.now()
		# curr_date = datetime.datetime(2025, 7, 4)
		if curr_date.month >= 8:
			self.curr_szn = f'{curr_date.year}-{str(curr_date.year + 1)[-2:]}'
		else:
			self.curr_szn = f'{curr_date.year - 1}-{str(curr_date.year)[-2:]}'

		self.validator = DataValidator(config)

	async def execute(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
		szn = ctx.get("season", self.curr_szn)
		gameweeks = ctx.get('gameweeks')  # None (all game weeks)
		if not gameweeks:
			df = await self._collect_merged_gws(szn)
		else:
			df = await self._collect_gws(gameweeks)
			ctx['data'] = df
			print('Context after data collection:', ctx)
			self.validator.execute(ctx)
		# await self._fetch_teams()
		return ctx

	async def _collect_merged_gws(self, season):
		data_url = f'{self.base_url}/{season}/gws/merged_gw.csv'
		async with ClientSession() as sess:
			async with sess.get(data_url) as res:
				data = await res.text()
				df = pl.read_csv(io.StringIO(data))

	async def _collect_gws(self, gameweeks: List[int]):
		base_data_url = f'{self.base_url}/{self.curr_szn}/gws'
		dfs = []
		async with ClientSession() as sess:
			tasks = []
			for gw in gameweeks:
				url = f'{base_data_url}/gw{gw}.csv'
				tasks.append(self._fetch_gw(sess, url, gw))

			# Gather all tasks (run all fetches asynchronously)
			results = await asyncio.gather(*tasks)
			for result in results:
				if isinstance(result, Exception):
					self.logger.warning(f"Failed to fetch gameweek: {result}")
					continue

				if result is not None:
					dfs.append(result)

		if dfs:
			combined_df = pl.concat(dfs)
			return combined_df

	async def _fetch_gw(self, sess: ClientSession, url: str, gw: int):
		try:
			async with sess.get(url) as res:
				data = await res.text()
				df = pl.read_csv(io.StringIO(data))
				# Using pl.lit to avoid ambiguity and ensure Polars expressions are being passed to with_cols
				df = df.with_columns(
					pl.lit(gw).alias('gw'),
					pl.lit(self.curr_szn).alias('season')
				)
				return df
		except Exception as e:
			print('EXCEPTION', e)
			return None

	async def _fetch_teams(self):
		data_url = f'{self.base_url}/{self.curr_szn}/teams.csv'
		async with ClientSession() as sess:
			async with sess.get(data_url) as res:
				data = await res.text()
				df = pl.read_csv(io.StringIO(data))
				return df


class FPLAPICollector(PipelineStep):
	def __init__(self, config):
		super().__init__("fpl_collector", config)
		self.base_url = config.get('sources', {}) \
			.get('fpl_base_url', 'https://fantasy.premierleague.com/api')
		self.logger = logging.getLogger(__name__)
		self.session = None
		self.validator = DataValidator(config)

	async def execute(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
		async with ClientSession() as sess:
			self.session = sess
			bootstrap_data = await self._fetch_bootstrap_data()
			ctx['bootstrap_data'] = bootstrap_data

		pass

	async def _fetch_transfer_history(self, _id: int):
		try:
			data_url = f'{self.base_url}/entry/{_id}/transfers'
			async with self.session.get(data_url) as res:
				data = await res.text()
				df = pl.read_csv(io.StringIO(data))
				# Validate df
				return data

		except Exception as e:
			print(f'TRANSFER FETCH EXCEPTION: {e}')
			self.logger.error("Failed to fetch transfer history!")
			raise


	async def _fetch_bootstrap_data(self):
		"""
		Fetches bootstrap data from the FPL API
		"""
		try:
			data_url = f'{self.base_url}/bootstrap-static'
			async with self.session as sess:
				async with sess.get(data_url) as res:
					data = await res.json()
					return data
		except Exception as e:
			self.logger.error(f'Failed to fetch bootstrap data: {e}')


async def _fetch_league_standings(self, _id: int, n_results: int):
	data_url = f'{self.base_url}/leagues-classic/{_id}/history/'
	try:
		async with ClientSession() as sess:
			async with sess.get(data_url) as res:
				data = await res.text()
				df = pl.read_csv(io.StringIO(data))
				# self.validator
				return df
	except Exception as e:
		self.logger.error(f'Failed to fetch league standings')
		raise


async def main():
	collector = VastaavCollector({
		'season': '2024-25',
		'gameweeks': [1,],
	})
	pipeline = Pipeline([
		collector,
	])
	await pipeline.execute()  


if __name__ == "__main__":
	asyncio.run(main())
