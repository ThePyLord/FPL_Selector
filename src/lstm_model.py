import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import tqdm
from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler

POSITION_FEATURES = {
	'GK': [
		'minutes', 'clean_sheets', 'saves', 'penalties_saved',
		'expected_goals_conceded', 'goals_conceded', 'influence',
		'ict_index', 'value', 'selected', 'transfers_balance', 'form',
		'team_form', 'consecutive_starts', 'assists', 'goals_scored'
	],
	'DEF': [
		'minutes', 'clean_sheets', 'goals_scored', 'assists',
		'expected_goals', 'expected_assists', 'threat', 'creativity',
		'influence', 'ict_index', 'value', 'selected', 'transfers_balance',
		'form', 'team_form', 'team_scoring_form',
		'points_trend', 'consecutive_starts', 'expected_goals_conceded'
	],
	'MID': [
		'minutes', 'goals_scored', 'assists', 'clean_sheets',
		'expected_goals', 'expected_assists', 'threat', 'creativity',
		'influence', 'ict_index', 'value', 'selected', 'transfers_balance',
		'form', 'team_form', 'team_scoring_form',
		'points_trend', 'consecutive_starts'
	],
	'FWD': [
		'minutes', 'goals_scored', 'assists', 'expected_goals',
		'expected_assists', 'threat', 'creativity', 'influence',
		'ict_index', 'value', 'selected', 'transfers_balance', 'form',
		'team_form', 'team_scoring_form',
		'points_trend', 'consecutive_starts'
	]
}


class FantasyFootballDataset(Dataset):
	def __init__(self, 
				 df: pd.DataFrame,
				 num_prev_gameweeks: int = 6,
				 position: str = 'MID',
				 target_col: str = 'total_points'):
		"""
		Create sequences for LSTM training
		
		Args:
			df: DataFrame with player gameweek data
			sequence_length: Number of previous gameweeks to use
			target_col: Column to predict
		"""
		df = df[df['position'] == position].copy()
		self.lookback = num_prev_gameweeks

		# Features to use in the model
		self.feature_cols = POSITION_FEATURES[position]

		# Prepare sequences
		self.sequences, self.targets = self._prepare_sequences(df, target_col)

	def _prepare_sequences(self, 
						  df: pd.DataFrame, 
						  target_col: str) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Convert DataFrame into sequences for each player"""
		sequences = []
		targets = []

		# Scale features
		scaler = StandardScaler()
		df[self.feature_cols] = scaler.fit_transform(df[self.feature_cols])

		# Create sequences for each player
		for name in df['name'].unique():
			player_data = df[df['name'] == name].sort_values('kickoff_time')

			# Get features
			features = player_data[self.feature_cols]

			if len(features) < self.lookback:
				continue

			# Create sequences
			for i in range(len(features) - self.lookback):
				seq = features[i:(i + self.lookback)] # 
				if i + self.lookback < len(player_data):
					target = player_data[target_col].iloc[i + self.lookback]
					sequences.append(seq.values)
					targets.append(target)
				# target = player_data[target_col].iloc[i + self.sequence_length]

				# sequences.append(seq)
				# targets.append(target)
		sequences = np.array(sequences)
		targets = np.array(targets)
		return torch.tensor(sequences), torch.tensor(targets)

	def __len__(self) -> int:
		return len(self.sequences)

	# def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
	# 	return (torch.FloatTensor(self.sequences[idx]),
	# 			torch.FloatTensor([self.targets[idx]]))
	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		seqs: torch.Tensor = self.sequences[idx]
		seqs = seqs.type(torch.float32)
		targets = torch.tensor([self.targets[idx]]).type(torch.float32)

		return seqs, targets

class PointsModel(nn.Module):
	def __init__(self, 
				 input_size: int,
				 hidden_size: int = 128,
				 num_layers: int = 2,
				 dropout: float = 0.2):
		super().__init__()
		
		self.lstm = nn.LSTM(
			input_size=input_size,
			hidden_size=hidden_size,
			num_layers=num_layers,
			dropout=dropout,
			batch_first=True
		)
		
		self.attention = nn.Sequential(
			nn.Linear(hidden_size, hidden_size),
			nn.Tanh(),
			nn.Linear(hidden_size, 1)
		)
		
		self.fc = nn.Sequential(
			nn.Linear(hidden_size, 32),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(32, 1)
		)
		
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# LSTM output
		lstm_out, _ = self.lstm(x)  # Shape: (batch, seq_len, hidden_size)

		# Attention weights
		attention_weights = self.attention(lstm_out)  # Shape: (batch, seq_len, 1)
		attention_weights = torch.softmax(attention_weights, dim=1)
		
		# Apply attention
		context = torch.sum(attention_weights * lstm_out, dim=1)
		
		# Final prediction
		out = self.fc(context)
		return out

class FPLLSTMPredictor:
	def __init__(self,
				 sequence_length: int = 5,
				 hidden_size: int = 64,
				 num_layers: int = 1,
				 learning_rate: float = 0.001,
				 batch_size: int = 32,
				 num_epochs: int = 50,
				 val_size: float = 0.2):
		self.sequence_length = sequence_length
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.val_size = val_size
		
	def train(self, train_df: pd.DataFrame) -> None:
		# Create dataset
		dataset = FantasyFootballDataset(train_df, self.sequence_length)
		dataloader = torch.utils.data.DataLoader(
			dataset, 
			batch_size=self.batch_size,
			shuffle=False
		)
		
		# Initialize model
		self.model = PointsModel(
			input_size=len(dataset.feature_cols),
			hidden_size=self.hidden_size,
			num_layers=self.num_layers
		)
		
		# Training setup
		criterion = nn.MSELoss()
		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
		
		# Loss values
		self.losses = []
		self.epochs = []

		# Training loop
		self.model.train()
		for epoch in tqdm.tqdm(range(self.num_epochs)):
		# for epoch in range(self.num_epochs):
			total_loss = 0
			
			for sequences, targets in dataloader:
				
				optimizer.zero_grad()
				outputs = self.model(sequences)
				
				loss = criterion(outputs, targets)
				loss.backward()
				self.losses.append(loss.item())
				self.epochs.append(epoch)
				optimizer.step()
				total_loss += loss.item()
			
			# if (epoch + 1) % 10 == 0:
			# 	print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {total_loss/len(dataloader):.4f}')
	
	def loss_epoch(self):
		return self.epochs, self.losses

	def predict(self, test_df: pd.DataFrame) -> np.ndarray:
		"""Make predictions for test data"""
		dataset = FantasyFootballDataset(test_df, self.sequence_length)
		dataloader = torch.utils.data.DataLoader(
			dataset,
			batch_size=self.batch_size,
			shuffle=False
		)
		
		self.model.eval()
		predictions = []
		aligned_targets = []
		
		with torch.no_grad():
			y_pred = self.model(dataset.sequences.float())
			
			for sequences, _ in dataloader:
				outputs = self.model(sequences)
				print(outputs)
				predictions.extend(outputs.numpy().flatten())
				aligned_targets.extend(_.numpy().flatten())
		
		return np.array(predictions), np.array(aligned_targets)
