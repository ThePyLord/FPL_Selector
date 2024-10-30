from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from utils import (
	fetch_data,
	fetch_transfer_history,
	fetch_gw_history, 
	fetch_standings,
	fetch_player_data,
	fetch_gw_data
)

# Data manipulation
import yaml
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning
from sklearn.model_selection import ( 
	KFold, 
	StratifiedGroupKFold, 
	cross_val_score,
	TimeSeriesSplit
)
from sklearn.ensemble import (
	RandomForestClassifier,
	RandomForestRegressor,
	GradientBoostingRegressor
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sns.set_theme(style='darkgrid')

# Create empty dataframe and store all gameweeks of the season
merged_df = pd.read_csv('data/2023-24.csv')
# for i in range(1, 39):
#     data = fetch_gw_data('2023-24', i)
#     data['GW'] = i
#     merged_df = pd.concat([merged_df, data])


# Sort by 'name' and 'kickoff_time' to ensure chronological order for each player
merged_df = merged_df.sort_values(by=["name", "kickoff_time"])

merged_df["kickoff_time"] = pd.to_datetime(merged_df["kickoff_time"])

# Define a function to calculate the form (average score in last few games or last 30 days)
def calculate_player_form(player_df):
    # Take the most recent N games (e.g., last 5 games)
    last_game = player_df.kickoff_time.max()

    # Filter games that are within the last 30 days
    last_30_days_games = player_df[
        player_df["kickoff_time"] >= (last_game - pd.Timedelta(days=30))
    ]

    # Calculate average points in those last 30 days
    form = (
        last_30_days_games["total_points"].mean().round(1)
        if len(last_30_days_games) > 0
        else 0
    )
    return form


# Group by player and calculate form based on their last 5 games (or you can change this to 30 days)
player_form = (
    merged_df.groupby("name", group_keys=False) #[['total_points', 'kickoff_time', 'GW']]
    .apply(lambda x: calculate_player_form(x))
    .reset_index()
)

# Rename the columns for clarity
player_form.columns = ["name", "form"]


# Merge the form with the original dataframe
merged_df = pd.merge(merged_df, player_form, on="name")
# merged_df.to_csv('data/2023-24.csv')
merged_df['kickoff_time'] = pd.to_datetime(merged_df['kickoff_time'])
print(f'{'form'} in merged_df.columns: {'form' in merged_df.columns}')
positions = merged_df.groupby(['position'])

numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
categorical_cols = merged_df.select_dtypes(include=[object]).columns

ts_cv = TimeSeriesSplit(n_splits=5)
X = merged_df.drop(['name', 'team'])
# Define the features and target variable
features = [
	"form",
	"total_points",
	"minutes",
	"goals_scored",
	"assists",
	"clean_sheets",
	"goals_conceded",
	"own_goals",
	"penalties_saved",
	"penalties_missed",
	"yellow_cards",
	"red_cards",
	"saves",
	"bonus",
	"bps",
	"influence",
	"creativity",
	"threat",
	"ict_index",
	"value",
	"transfers_balance",
	"selected"
]

X = merged_df[features]
y = merged_df["total_points"]
for train_idx, test_idx in ts_cv.split(X):
	X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
	y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

	# Create a pipeline with a StandardScaler and a RandomForestRegressor
	pipeline = make_pipeline(StandardScaler(), RandomForestRegressor())

	# Fit the model on the training data
	pipeline.fit(X_train, y_train)

	# Predict the target variable on the test data
	y_pred = pipeline.predict(X_test)

	# Calculate the mean squared error
	mse = mean_squared_error(y_test, y_pred)
	print(f"Mean Squared Error: {mse}")