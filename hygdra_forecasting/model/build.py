import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvCausalLTSM(nn.Module):
    """
    convolutionnal LTSM class specialized to forecast causal time series

    parameters :
        input_shape (tuple[int]): dimension shape of the imput (batch, width, height) or (width, height)
        dropout (float): from 0 to 1 random dropout applyed on layer to minimise overfitting
    """
    def __init__(self, input_shape, dropout:float=0.2):
        super(ConvCausalLTSM, self).__init__()
        
        self.conv1d = nn.Conv1d(in_channels=input_shape[0], out_channels=64, kernel_size=5, padding=2)  # padding='same' equivalent
        self.dropout = nn.Dropout(dropout)
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, dropout=dropout, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, dropout=dropout, bidirectional=False)
        self.lstm3 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, dropout=dropout, bidirectional=False)
        self.lstm4 = nn.LSTM(input_size=64, hidden_size=16, num_layers=2, batch_first=True, dropout=dropout, bidirectional=False)

        # Linear output layer
        self.fc = nn.Linear(16, 1)  # For a single regression output

    def forward(self, x):
        """
        give the forecasting ouput of the x input based on model training

        arg :
            x (tensor): ND tensor of shape the (batch, input_shape)
        return :
            tensor : predicted output of shape (batch, output)
        """
        # Conv1D layer
        x = self.conv1d(x)
        x = self.dropout(x)
        x = self.maxpool(x)

        # Rearrange to (batch_size, seq_len, features) format for LSTM
        x = x.transpose(1, 2)  # Shape: (batch_size, seq_len, features)

        # LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)

        # Take the output from the last LSTM time step
        x = x[:, -1, :]  # Shape: (batch_size, 16)

        # Fully connected output layer
        x = self.fc(x)

        return x
    
class LtsmAttentionforecastPred(nn.Module):
    """
    Optimized GraphForecast Model for Stock Prediction

    Parameters:
        input_shape (tuple[int]): (channels, sequence_length)
        dropout (float): Dropout rate for regularization
    """
    def __init__(self, input_shape, dropout: float = 0.2):
        super(LtsmAttentionforecastPred, self).__init__()

        # Convolutional feature extractor
        self.conv1 = nn.Conv1d(in_channels=input_shape[0], out_channels=64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(50)  # Adaptive pooling for dynamic input

        # BiLSTM for sequential modeling
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, 
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.layer_norm = nn.LayerNorm(256)

        # Attention mechanism
        self.attn = nn.Linear(256, 1)  # Learnable attention weights
        self.softmax = nn.Softmax(dim=1)

        # Fully connected output
        self.fc = nn.Linear(256, 1)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        """Forward pass for time-series forecasting"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch_norm1(x)
        x = self.dropout(x)
        x = self.pool(x)

        x = x.transpose(1, 2)  # Convert to (batch, seq_len, features) for LSTM

        x, _ = self.lstm(x)
        x = self.layer_norm(x)

        # Attention mechanism
        attn_weights = self.softmax(self.attn(x))  # (batch, seq_len, 1)
        x = torch.sum(x * attn_weights, dim=1)  # Weighted sum of LSTM outputs

        x = self.fc(x)
        x = self.activation(x)

        return x