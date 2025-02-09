import torch
from liquidnet.main import LiquidNet
from liquidnet.vision_liquidnet import VisionLiquidNet
import torch.nn as nn


class ConvCausalLiquid(nn.Module):
    """
    convolutionnal LTSM class specialized to forecast causal time series

    parameters :
        input_shape (tuple[int]): dimension shape of the imput (batch, width, height) or (width, height)
        dropout (float): from 0 to 1 random dropout applyed on layer to minimise overfitting
    """
    def __init__(self, input_shape, batch_size=128, dropout:float=0.2):
        super(ConvCausalLiquid, self).__init__()
        
        self.conv1d = nn.Conv1d(in_channels=input_shape[0], out_channels=64, kernel_size=5, padding=2)  # padding='same' equivalent
        self.dropout = nn.Dropout(dropout)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        num_units = 128
        # LSTM layers
        self.ltc_cell = LiquidNet(num_units)

        # Linear output layer
        self.fc = nn.Linear(16, 1)  # For a single regression output
        self.initial_state = torch.zeros(batch_size, num_units)

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

        x, self.initial_state = ltc_cell(inputs, self.initial_state)

        # Fully connected output layer
        x = self.fc(x)

        return x


# model = VisionLiquidNet(64, 10) -> graph network