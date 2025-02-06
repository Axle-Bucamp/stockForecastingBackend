import torch.nn as nn

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

# Example usage:
# Assuming input shape is (batch_size, channels, sequence_length), e.g., (batch_size, 7, 35)
if __name__ == '__main__':
    input_shape = (7, 35)  # Channels=7, Sequence length=35
    model = ConvCausalLTSM(input_shape)
