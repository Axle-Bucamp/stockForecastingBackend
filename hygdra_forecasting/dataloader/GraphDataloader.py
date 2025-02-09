from hygdra_forecasting.utils.preprocessing import ohlv_to_dataframe, dataframe_to_graph_dataset
from torch.utils.data import Dataset, DataLoader
from torch import tensor, float32

class StockGraphDataset(Dataset):
    def __init__(self, ticker: list[str], indics: list[str], period:str="2y", interval:str='1d'):
        """
        Args:
            ticker (str): The ticker symbol for the stock (e.g., 'AAPL').
            start_date (str): Start date for the stock data (e.g., '2020-01-01').
            end_date (str): End date for the stock data (e.g., '2021-01-01').
            seq_length (int): The length of the input sequence for each sample (e.g., 60 for 60 days).
            features (List[str]): The list of features to include from the data (e.g., 'Close', 'Open', etc.).
        """
        self.ticker = ticker 
        self.indics = indics
        self.period = period
        self.interval = interval

        # Fetch stock data
        self.data, self.label = self.fetch_data()

    def fetch_data(self):
        """
        Fetch stock data from Yahoo Finance
        """
        df, label = ohlv_to_dataframe(self.ticker, self.period, self.interval)
        df_indic, _ = ohlv_to_dataframe(self.indics, self.period, self.interval)
        train_sequences, train_labels = dataframe_to_graph_dataset(df, df_indic, label, self.ticker)
        return train_sequences, train_labels

    def __len__(self):
        """
        Return the total number of samples (sequences) in the dataset.
        """
        return len(self.label)

    def __getitem__(self, index):
        """
        Generate a sequence of data for the given index.
        """

        # Convert numpy arrays to PyTorch tensors
        x = tensor(self.data[index], dtype=float32)
        y = tensor([self.label[index]], dtype=float32)
        
        return x, y

if __name__ == '__main__':
    # Example usage
    ticker = 'AAPL'

    # Create the dataset and DataLoader
    dataset = StockGraphDataset(ticker=[ticker], indics=["AAPL"])
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)

    # Example: Iterate through the DataLoader
    for batch_idx, (x, y) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1} - X: {x.shape}, Y: {y.shape}")
        # Now you can pass the data (x) and labels (y) to your model
        break  # Just for demonstration, break after one batch
