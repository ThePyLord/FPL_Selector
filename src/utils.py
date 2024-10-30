# Standard imports
import urllib.request as request
import logging
# Data manipulation
import json, yaml
import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(
	level=logging.INFO, 
	format='%(levelname)s %(asctime)s - %(message)s',
	datefmt='%d-%b %H:%M')

class Database:
	def __init__(self, path="../config.yaml"):
		self.path = path
		self.engine = self.make_connection()

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
		params = {col: val for col, val in zip(column_lst, values)}
		with self.engine.connect() as conn:
			conn.execute(text(query), params)
			conn.commit()

	def get_data(self, query, params=None):
		"""Fetches data from the database
		Args:
						query: str, SQL query
						params: tuple, parameters to pass to the query
		Returns:
						df, dataframe of the fetched data
		"""
		with self.engine.connect() as conn:
			print(f"Query: {query}")
			df = pd.read_sql(text(query), conn, params=params)
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
	# df.to_sql(table_name, engine, if_exists='replace', index=False)
	min_year = pd.to_datetime(df['kickoff_time']).min().year
	columns = df.columns.tolist()
	existing_row_count = db.get_data(f"SELECT COUNT(*) FROM {table_name} WHERE season = '{min_year}-{str(min_year + 1)[2:]}'")
	row_count = 0
	for _, row in df.iterrows():
		# print(row.values)
		# print('Retrieving row data')
		row_from_db = db.get_data(
			f"""
			SELECT * FROM {table_name}
			WHERE name = %s AND kickoff_time = %s AND GW = %s
			""", (row['name'], row['kickoff_time'], row['GW']))
		
		season = f'{min_year}-{str(min_year + 1)[2:]}'
		if row_from_db is not None:
			row_count += 1
			# print(f'Year: {season}')
			# print(f"Row already exists: {row['name']}, {row['kickoff_time']}, {row['GW']}")
			continue
		db.write_to_sql(f'{table_name}', columns, row.values.tolist().append(season))
	print(f"{row_count} rows already exist in the database.")