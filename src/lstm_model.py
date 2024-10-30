import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler

class FPLSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 df: pd.DataFrame,
                 sequence_length: int = 6,
                 target_col: str = 'total_points'):
        """
        Create sequences for LSTM training
        
        Args:
            df: DataFrame with player gameweek data
            sequence_length: Number of previous gameweeks to use
            target_col: Column to predict
        """
        self.sequence_length = sequence_length
        
        # Features to use in the model
        self.feature_cols = [
            'minutes', 'goals_scored', 'assists', 'clean_sheets',
            'expected_goals', 'expected_assists', 'threat', 'creativity',
            'influence', 'ict_index', 'value', 'selected',
            'transfers_balance', 'form'  # form should be pre-calculated
        ]
        
        # Prepare sequences
        self.sequences, self.targets = self._prepare_sequences(df, target_col)
        
    def _prepare_sequences(self, 
                          df: pd.DataFrame, 
                          target_col: str) -> Tuple[np.ndarray, np.ndarray]:
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
            features = player_data[self.feature_cols].values
            
            # Create sequences
            for i in range(len(features) - self.sequence_length):
                seq = features[i:(i + self.sequence_length)]
                target = player_data[target_col].iloc[i + self.sequence_length]
                
                sequences.append(seq)
                targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.FloatTensor(self.sequences[idx]),
                torch.FloatTensor([self.targets[idx]]))

class FPLLSTM(nn.Module):
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 64,
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
                 sequence_length: int = 6,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 num_epochs: int = 50):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
    def train(self, train_df: pd.DataFrame) -> None:
        # Create dataset
        dataset = FPLSequenceDataset(train_df, self.sequence_length)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Initialize model
        self.model = FPLLSTM(
            input_size=len(dataset.feature_cols),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for sequences, targets in dataloader:
                optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Make predictions for test data"""
        dataset = FPLSequenceDataset(test_df, self.sequence_length)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for sequences, _ in dataloader:
                outputs = self.model(sequences)
                predictions.extend(outputs.numpy().flatten())
                
        return np.array(predictions)
