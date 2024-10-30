from time import time
import pandas as pd
import numpy as np
from sqlalchemy import text
from utils import Database, dataframe_to_db
from feature_engineering import FPLFeatureEngineering
# import matplotlib.pyplot as plt


var = 5
cols = [
    "name",
    "position",
    "team",
    "xP",
    "assists",
    "bonus",
    "bps",
    "clean_sheets",
    "creativity",
    "element",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals",
    "expected_goals_conceded",
    "fixture",
    "goals_conceded",
    "goals_scored",
    "ict_index",
    "influence",
    "kickoff_time",
    "minutes",
    "opponent_team",
    "own_goals",
    "penalties_missed",
    "penalties_saved",
    "red_cards",
    "round",
    "saves",
    "selected",
    "starts",
    "team_a_score",
    "team_h_score",
    "threat",
    "total_points",
    "transfers_balance",
    "transfers_in",
    "transfers_out",
    "value",
    "was_home",
    "yellow_cards",
    "GW",
]

vals = [
    "Alex Scott",
    "MID",
    "Bournemouth",
    1.6,
    0,
    0,
    11,
    0,
    12.8,
    77,
    0.01,
    0.01,
    0.0,
    1.02,
    6,
    1,
    0,
    3.6,
    22.8,
    "2024-08-17T14:00:00Z",
    62,
    16,
    0,
    0,
    0,
    0,
    1,
    0,
    4339,
    1,
    1,
    1,
    0.0,
    2,
    0,
    0,
    0,
    50,
    False,
    0,
    1
]

db = Database('../config.yaml')
df = pd.read_csv('https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/gws/merged_gw.csv')

# db.write_to_sql('gameweek_stats', cols, vals)
class Timer:
	def __init__(self):
		self.start = time()

	def __enter__(self):
		return self

	def __exit__(self, *args):
		self.end = time()
		self.interval = self.end - self.start
		print(f"Time taken: {self.interval:.2f} seconds")

query = """
SELECT * FROM gameweek_stats
WHERE season = '2023-24' AND position = 'MID'
"""

with Timer() as t:
    dataframe_to_db(df, 'gameweek_stats', db)
    

with Timer() as t:
	df = db.get_data(query)
	

# print(df.head())
# print(f'Type of kickoff_time: {df['kickoff_time'].dtype}')
feat_eng = FPLFeatureEngineering()
# df = feat_eng.transform(df)
# print(f'{df.columns}')