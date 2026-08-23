from time import time
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import torch
import statsmodels.api as sm
import statsmodels.tsa.arima as arima
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import torch.utils
import yaml
from utils import Database, Timer, setup_logging
from feature_engineering import FplPreprocessor
from lstm_model import PointsModel, FPLLSTMPredictor, FantasyFootballDataset
import matplotlib.pyplot as plt

setup_logging()


db = Database('../config.yaml')
# url = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/2022-23/gws/merged_gw.csv'
# df = pd.read_csv(url)
# db.write_to_sql('gameweek_stats', cols, vals)

# dataframe_to_db(df, 'gameweek_stats', db)

with open('../config/features.yaml', 'r') as f:
	config = yaml.safe_load(f)
	features = config['position_features']
	target = config['target']

query = """
SELECT * FROM gameweek_stats
WHERE season < '2023-24' AND position = 'MID'
ORDER BY kickoff_time;
"""


with Timer() as t:
	df = db.get_data(query)


feat_eng = FplPreprocessor(rolling_windows=[5], lag_features=['total_points'])

df = feat_eng.transform(df).dropna()

midfielders = df[df.position == 'MID']
train_df = df[df.gw <= 25]
test_df = df[df.gw > 25]


# LSTM Model
predictor = FPLLSTMPredictor(
	sequence_length=7,
	hidden_size=64,
	num_layers=2,
	learning_rate=0.001,
	batch_size=32,
	num_epochs=100,
)

predictor.train(train_df)
test_ds = FantasyFootballDataset(
	test_df, num_prev_gameweeks=6, position='MID', target_col='total_points'
)

predictions, outs = predictor.predict(test_df)
print(f'Predictions: {predictions.shape}')                                                                                                                                           
# exit()

print(f'RMSE: {np.sqrt(mean_squared_error(test_df, predictions))}')

# ARIMA Model
dta = midfielders.query('name == "Mohamed Salah"').set_index('kickoff_time')
dta.drop(columns=['name', 'position', 'team', 'opponent_team', 'season'], inplace=True)
dta.sort_index(inplace=True)

# train_df = dta[dta.gw <= 25]
# test_df = dta[dta.gw > 25]
# Order of ARIMA model is (p, d, q)
# d = 0 (no differencing)
model = arima.model.ARIMA(train_df['total_points'], order=(2, 0, 1))
model_fit = model.fit()
print(model_fit.summary())

forecast = model_fit.forecast(steps=len(test_df))

print(f'RMSE: {np.sqrt(mean_squared_error(test_df["total_points"], forecast))}')

# OLS Model
X = train_df[features['mid']]
X = sm.add_constant(X)
y = train_df[target]
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

predictions = results.predict(sm.add_constant(test_df[features['mid']]))
print(f'RMSE: {np.sqrt(mean_squared_error(test_df[target], predictions))}')

plt.subplot(2, 1, 1)
plt.plot(test_df.index, test_df[target], label='Actual', color='blue')
plt.plot(test_df.index, predictions, label='Predicted', color='red')
plt.legend()
plt.title('OLS Predictions')

plt.subplot(2, 1, 2)
plt.plot(test_df.index, test_df[target], label='Actual', color='blue')
plt.plot(test_df.index, forecast, label='Predicted', color='red')
plt.legend()
plt.title('ARIMA Predictions')
plt.show()
