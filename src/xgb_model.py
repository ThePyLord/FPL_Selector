import yaml

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from feature_engineering import FplPreprocessor
from utils import Database


sns.set_theme(style='darkgrid')

with open("../config/features.yaml", "r") as f:
    config = yaml.safe_load(f)
    features = config["position_features"]
    target = config["target"]

db = Database("../config.yaml")

position = features['mid']

df = db.get_data("""
	SELECT * FROM gameweek_stats
	WHERE season = '2022-23'
""")

print(f'Retrieving data from {df.loc[0, "season"]} season')
feat_eng = FplPreprocessor([5])
df = df[df.position == 'MID']
df = feat_eng.transform(df)

# Regressor
regressor = xgb.XGBRegressor(n_estimators = 400, random_state = 42)
train = df[df.gw <= 25]
test = df[df.gw > 25]
# Train-Test split
X_train = train[features['mid']]
y_train = train[target]

X_test = test[features['mid']]
y_test = test[target]

regressor.fit(X_train, y_train)

predictions = regressor.predict(X_test)

loss = mean_squared_error(y_test, predictions)
print(f'Loss on XGBRegressor: {loss}')


# Test on 2022-23 season
df = db.get_data("""
	SELECT * FROM gameweek_stats
	WHERE season = '2023-24'
""")
print(f'Retrieving data from {df.loc[0, "season"]} season')
df = df[df.position == 'MID']
df = feat_eng.transform(df)
X = df[features['mid']]
y = df[target]
predictions = regressor.predict(X)
loss = mean_squared_error(y, predictions)
print(f'Loss on 2023-24 season: {loss}')

# Select a random gameweek and a random player from that gameweek
player_info = db.get_data("""
    WITH random_gw AS (
        SELECT floor(random() * 25 + 1)::int AS gw
    ),
    random_player AS (
        SELECT name
        FROM gameweek_stats
        WHERE gw = (SELECT gw FROM random_gw)
        AND position = 'MID'
        ORDER BY random()
        LIMIT 1
    )
    SELECT 
        (SELECT gw FROM random_gw) AS gw,
        (SELECT name FROM random_player) AS name
""")

random_gw = player_info.loc[0, 'gw']
print(f"SELECTED RANDOM GAMEWEEK: {random_gw}")
player_name = player_info.loc[0, 'name']

print(f"PREDICTING POINTS FOR: {player_name} in gameweek {random_gw + 1}")

# Retrieve data up to the selected gameweek for the player
player_data = db.get_data(f"""
    SELECT *
    FROM gameweek_stats
    WHERE gw <= {random_gw}
    AND position = 'MID'
    AND name = '{player_name}'
""")

# Apply feature engineering
player_ft = feat_eng.transform(player_data)
player_ft = player_ft[features['mid']]

# Predict the points for the next gameweek
player_pred = np.floor(regressor.predict(player_ft))

# Print the prediction
print(f"Predicted points for {player_name} in gameweek {random_gw + 1}: {player_pred[-1]}")

# Plot feature importance
feature_imp = regressor.feature_importances_
indices = feature_imp.argsort()[::-1]
important_features = X_train.columns[indices][:10]
# sns.barplot(x = X_train.columns[indices], y = feature_imp[indices])
# plt.title('Feature Importance')
# plt.xlabel('Features')
# plt.ylabel('Importance')
# plt.xticks(rotation=45)
# plt.show()
