# Standard imports
from time import time
import urllib.request as request
import tempfile
import logging
# Data manipulation
import json, yaml
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError

def setup_logging():
    logging.basicConfig(
		level=logging.INFO, 
		format='%(levelname)s %(asctime)s - %(message)s',
		datefmt='%d-%b %H:%M')


class Timer:
    def __init__(self):
        self.start = time()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.end = time()
        self.interval = self.end - self.start
        print(f"Time taken: {self.interval:.2f} seconds")


class Database:
	def __init__(self, path="../config.yaml"):
		self.path = path
		try:
			self.engine = self.make_connection()
		except Exception as e:
			logging.error(f"Error connecting to the database: {e}")

	def make_connection(self):
		"""Creates a connection to the database
		Returns:
			engine, connection to the database
		"""
		path = self.path if self.path else "../config.yaml"
		with open(path, "r") as f:
			config = yaml.safe_load(f)
			db_cfg = config['db']
			engine = create_engine(f'postgresql://{db_cfg["host"]}:{db_cfg["port"]}/{db_cfg["database"]}')
			logging.info("Connected to the database.")
		return engine

	def write_to_sql(self, table: str, column_lst: list[str], values: list[str]):
		""" Write a row of data to a table.
		Args:
			table: str, the name of the table to insert into.
			column_lst: list[str], the column names of the table.
			values: str, the values to be inserted into the table """
		columns = ', '.join(column_lst)
		# placeholders = ', '.join(['%s'] * len(column_lst))
		placeholders = ", ".join([f":{col}" for col in column_lst])
		query = (f'INSERT INTO {table} ({columns}) VALUES ({placeholders})'
		)
		print(f'params: {values}')
		params = {col: val for col, val in zip(column_lst, values)}
		with self.engine.connect() as conn:
			try:
				conn.execute(text(query), params)
				conn.commit()
			except Exception as e:
				logging.error(f"Error writing to the database: {e}")
				conn.rollback()


	def bulk_insert(self, table: str, df: pd.DataFrame):
		"""Bulk insert data into the database
		Args:
			table: str, the name of the table to insert into.
			df: pd.DataFrame, the data to be inserted into the table
		"""
		min_year = pd.to_datetime(df["kickoff_time"]).min().year
		df_cpy = df.copy()
		df_cpy['season'] = f"{min_year}-{str(min_year + 1)[2:]}"
		records = df_cpy.to_dict(orient="records")
		with self.engine.connect() as conn:
			for row in records:
				inserts = insert(table).values(row)

				do_nothing_stmt = inserts.on_conflict_do_nothing(index_elements=['name', 'kickoff_time', 'gw', 'season'])

				try:
					# conn.execute(insert(table).values(records
					conn.execute(do_nothing_stmt)
					conn.commit()
				except IntegrityError as ie:
					logging.error(f"Error writing to the database: {ie}")
					conn.rollback()
				except Exception as e:
					logging.error(f"Error writing to the database: {e}")
					conn.rollback()
			conn.execute("""REINDEX TABLE gameweek_stats""")
				

	def get_data(self, query, params=None):
		"""Fetches data from the database
		Args:
						query: str, SQL query
						params: tuple, parameters to pass to the query
		Returns:
						df, dataframe of the fetched data
		"""
		with self.engine.connect() as conn:
			df = pd.read_sql(text(query), conn, params=params)
		if df.columns.str.contains("kickoff_time").any():
			df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
		return df


async def fetch_data(url):
	with request.urlopen(url) as response:
		return json.loads(response.read().decode())


async def fetch_transfer_history(ic):
	"""Fetches the transfer history of a given FPL manager's ID
	Args:
			ic: int, FPL manager's ID
	Returns:
			dict, transfer history of the FPL manager"""
	url = f"https://fantasy.premierleague.com/api/entry/{ic}/transfers/"
	return await fetch_data(url)


async def fetch_gw_history(ic):
	"""Fetches the gameweek history of a given FPL manager's ID
	Args:
			ic: int, FPL manager's ID
	Returns:
			dict, gameweek history of the FPL manager"""
	url = f"https://fantasy.premierleague.com/api/entry/{ic}/history/"
	return await fetch_data(url)


async def fetch_standings(_id: int = 314, n_results: int = 10):
	"""
	Fetches the standings of a given league's ID
	Args:
			_id: int, league id, default is 314 (Overall)
			n_results: int, number of results to return, default is 10
	Returns:
			dict, League standings
	"""
	url = f"https://fantasy.premierleague.com/api/leagues-classic/{_id}/standings/"
	result = await fetch_data(url)
	return result["standings"]["results"][:n_results]


async def fetch_player_data(_id: int):
	"""Fetches the data of a given player's ID
	Args:
			id: int, player's ID
	Returns:
			dict, player's data
	"""
	url = f"https://fantasy.premierleague.com/api/element-summary/{_id}/"
	return await fetch_data(url)


def fetch_gw_data(season: str, gw: int):
	"""Fetches the data of a given gameweek
	Args:
			season: str, season, e.g. 2020-21
			gw: int, gameweek number
	Returns:
			dict, gameweek data
	"""
	#     url = f"https://fantasy.premierleague.com/api/fixtures/?event={gw}&_={season}"
	try:
		if gw < 1 or gw > 38:
			raise ValueError("Gameweek should be between 1 and 38")
		if int(season[:4]) < 2016:
			raise ValueError("Season should be 2016 onwards")
		elif int(season[2:4]) > int(season[5:]):
			raise ValueError("Season should be in the format 'yyyy-yy' and the second year should be greater than the first")
		url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/{season}/gws/gw{gw}.csv"
		df = pd.read_csv(url)
		return df
	except ValueError as e:
		print(e)
		return None


def dataframe_to_db(df, table_name, db: Database):
	min_year = pd.to_datetime(df["kickoff_time"]).min().year
	columns = df.columns.tolist()
	columns.append("season")
	columns = list(map(str.lower, columns))
	existing_row_count = db.get_data(
		f"SELECT COUNT(*) FROM {table_name} WHERE season = '{min_year}-{str(min_year + 1)[2:]}'"
	)
	row_count = 0
	season = f"{min_year}-{str(min_year + 1)[2:]}"
	for _, row in df.iterrows():
		# print(row.values)
		# print("Retrieving row data")
		with db.engine.connect() as conn:
			row_from_db = conn.execute(
				text(
					f"""
					SELECT * FROM {table_name}
					WHERE name = :name 
					AND kickoff_time = :kickoff_time
					AND gw = :GW 
					AND season = :season
					"""
				),
				{
					"name": row["name"], 
					'kickoff_time': row["kickoff_time"],
					"GW": row["GW"], 
					"season": season
				},
			)
			row_from_db = row_from_db.fetchone()

		if row_from_db is not None:
			row_count += 1
			continue

		print('Inserting row')
		values = row.values.tolist()
		values.append(season)

		db.write_to_sql(table_name, columns, values)
	res = db.get_data(
		f"SELECT COUNT(*) FROM {table_name} WHERE season = '{min_year}-{str(min_year + 1)[2:]}'"
	)
	print(f"{res.values[0][0]} rows inserted into the database.")


def insert_gameweek_stats(df, table_name, db: Database):
	# Add season column if needed
	min_year = pd.to_datetime(df["kickoff_time"]).min().year
	df["season"] = f"{min_year}-{str(min_year + 1)[2:]}"
	rows = df.to_dict(orient="records")
	print(rows[0])
