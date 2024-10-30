# Creates a model to learn the strategy of the top 10 FPL managers

# Std lib
import asyncio
from functools import reduce
from pprint import pprint
from utils import (
	fetch_data,
	fetch_transfer_history,
	fetch_gw_history, 
	fetch_standings
)

# Data manipulation
import yaml
import pandas as pd
import numpy as np

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

with open('config.yaml') as file:
	config = yaml.load(file, Loader=yaml.FullLoader)

_id = config['my_id']
url = f"https://fantasy.premierleague.com/api/entry/{_id}/"
print(_id)


async def main():
	data = await fetch_data(url)
	classic_leagues = data['leagues']['classic']

	overall_league = reduce(lambda x, y: x if x['name'] == 'Overall' else y, classic_leagues)
	standings_uri = f'https://fantasy.premierleague.com/api/leagues-classic/{overall_league["id"]}/standings/'
	standings = await fetch_data(standings_uri)
	standings = standings['standings']['results']
	# pprint(standings)
	df = pd.DataFrame(standings[:10])
	print(df)
	transfer_hist = []
	for i, row in df.iterrows():
		ic = row['entry']
		# print(f"{row['entry_name']}'s id: {row['entry']}")
		transfers = await fetch_transfer_history(ic)
		transfer_hist.append(transfers)
	transfer_hist = sorted(transfer_hist, key = lambda x: len(x), reverse=True)
	# print([len(x) for x in transfer_hist])
	last_transfer = pd.DataFrame(transfer_hist[-1])
	pprint(last_transfer)
	print(f"ID's are unique: {df.id.nunique() == len(df)}")

asyncio.run(main())


