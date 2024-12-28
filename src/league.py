import asyncio
import yaml
from utils import fetch_gw_history, fetch_standings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

with open('../config/league.yaml') as f:
	config = yaml.safe_load(f)
	players = config['players']

def plot_point_progress(data):
    fig = px.line(
        data,
        x="gw",
        y="total_points",
        color="name",
        markers=True,
        title="FPL Points per Gameweek",
        hover_data=["points"],
    )

    fig.update_layout(
        xaxis_title="Gameweek",
        yaxis_title="Points",
        legend_title="Player",
        title={"xanchor": "center", "x": 0.5, "y": 0.95},
    )
    fig.show()

async def main():
	df = pd.DataFrame()
	for player in players:
		# print(f'Player: {player["name"]}, ID: {player["id"]}')
		standings = await fetch_gw_history(player['id'])
		standings = pd.DataFrame(standings['current'])
		standings['name'] = player['name']
		standings = standings[['name', 'event', 'points', 'total_points']]
		standings.rename(columns={'event': 'gw'}, inplace=True)
		df = pd.concat([df, standings])

	# goodness = await fetch_gw_history(5621308)
	print(df.shape)
	df['name'] = df['name'].str.capitalize()

	best_pos = df.groupby(['gw'], as_index=False)['points'].max()
	print(best_pos)

	# Add a column for the difference in points between each gameweek from the highest ranked player
	# plot the points of each player
	plot_point_progress(df)

asyncio.run(main())
