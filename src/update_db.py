### This script is used to update the database with the latest data from the vaastav/Fantasy-Premier-League repository.
### The data is read from a CSV file and inserted into the database using the dataframe_to_db function.
### Potential improvements/updates:
### - Add logging to track the progress of the script
### - Add error handling to handle exceptions and errors
### - Update the backup table before inserting the new data

import logging
import yaml
import pandas as pd
from utils import Database, Timer, dataframe_to_db, setup_logging

setup_logging()
db = Database('../config.yaml')
url = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/2024-25/gws/merged_gw.csv'
df = pd.read_csv(url)
df['kickoff_time'] = pd.to_datetime(df['kickoff_time'], utc = True)
# df['kickoff_time'] = df['kickoff_time'].dt.tz_convert('Europe/London')
print(f'Dataset size: {df.shape}')

dataframe_to_db(df, 'gameweek_stats', db)