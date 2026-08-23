from typing import Optional, Dict, Any, List

import sqlalchemy
import yaml
from matplotlib.table import table
from pexpect.pxssh import ExceptionPxssh
from sqlalchemy import create_engine, text, inspect, Table, MetaData, Inspector
from sqlalchemy.dialects import postgresql
import polars as pl
import logging

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError


class DatabaseManager:
	def __init__(self, config_path: str = "../config.yaml"):
		# Initialize logger
		self._init_logging()
		self.logger = logging.getLogger(__name__)

		self.config = self.load_config(config_path)
		self.engine: Optional[sqlalchemy.Engine]  = None

	def _init_logging(self):
		"""Initializes logging config"""
		logging.basicConfig(
			level=logging.INFO,
			format="%(levelname)s %(asctime)s - %(message)s",
			datefmt="%d-%b %H:%M",
		)

	def load_config(self, config_path: str):
		try:
			with open(config_path) as f:
				config = yaml.safe_load(f)
				return config.get("db", {})
		except FileNotFoundError as fnfe:
			self.logger.fatal(f"Config file {config_path} not found")

	def _create_engine(self):
		try:
			url = f"postgresql://{self.config['host']}:{self.config['port']}/{self.config['database']}"
			self.engine = create_engine(
				url,
				echo=False,
				pool_size=5,
				max_overflow=100,
				pool_pre_ping=True,
				pool_recycle=3600,
			)

			with self.engine.connect() as conn:
				conn.execute(text("SELECT 1"))
			self.logger.info("Successfully connected to database")
			return self.engine
		except Exception as e:
			self.logger.fatal("Failed to connect to database!")
			raise


	def check_table_existence(self, table_name: str) -> bool:
		try:
			inspector: Inspector = inspect(self.engine)
			if not inspector.has_table(table_name):
				self.logger.warning(f'Table {table_name} does not exist')
				return False
			return True
		except Exception as e:
			self.logger.error(f'Error checking table existence {e}')


	def table_info(self, table_name):
		try:
			insp: Inspector = inspect(self.engine)
			if not insp.has_table(table_name):
				return None

			cols = insp.get_columns(table_name)
			pks = insp.get_pk_constraint(table_name)
			return {
				'columns': cols,
				'primary_keys': pks
			}
		except Exception as e:
			self.logger.error(f'Error getting table info: {e}')
			return None

	def execute_query(self, query: str, params: Optional[Dict[str, Any]]):
		try:
			with self.engine.connect() as conn:
				df = pl.read_database(
					text(query),
					conn,
					execute_options={'parameters': params},
					schema_overrides={'kickoff_time': pl.Datetime}
				)
				datetime_cols = df.select(pl.col(pl.Datetime)).columns
				for col in datetime_cols:
					df[col].dt.replace_time_zone(time_zone='UTC')

				return df
		except Exception as e:
			self.logger.error(f'Failed to execute query {e}')
			raise


	def bulk_upsert(self, table_name: str, df: pl.DataFrame, conflict_columns: List[str]):
		"""
		Bulk upsert data from a DataFrame into a table
		Args:
			table_name: the table name
			df: the dataframe that will be upserted into the table

		Returns:

		"""
		if 'kickoff_time' in df.columns:
			df = self._add_season_feature(df.clone())
		with self.engine.connect() as conn:
			insp = inspect(self.engine)
			inserted, updated, skipped = 0, 0, 0
			records = df.to_dict(as_series=False)
			for record in records:
				try:
					stmt = insert(table_name).values(record)
					if conflict_columns:
						update_cols = {
							col: stmt.excluded[col]
							for col in record.keys()
							if col not in conflict_columns
						}
						stmt = stmt.on_conflict_do_update(
							index_elements=conflict_columns,
							set_=update_cols
						)
						updated += 1
					else:
						stmt = stmt.on_conflict_do_nothing()
						skipped += 1
					conn.execute(stmt)
					inserted += 1
				except IntegrityError as ie:
					self.logger.error(f'Integrity error for record {ie}')
					skipped += 1
					continue
				except Exception as e:
					self.logger.error(f'Exception inserting record: {e}')
					raise

				# Commit all changes
				conn.commit()

				# Reindex
				conn.execute(text(f'REINDEX TABLE {table_name}'))
				conn.commit()
		return inserted, updated, skipped


	def _add_season_feature(self, df: pl.DataFrame):
		df['kickoff_time'] = df['kickoff_time'].dt.replace_time_zone('UTC')

		start_time = df['kickoff_time'].min()

		if start_time.month >= 8:
			season = f'{start_time.year}-{str(start_time.year + 1)[-2:]}'
		else:
			season = f'{start_time.year - 1}-{str(start_time.year)[-2:]}'

		df['season'] = season
		return df


