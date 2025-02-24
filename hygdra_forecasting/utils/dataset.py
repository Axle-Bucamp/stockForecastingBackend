import numpy as np
import yfinance as yf
import pandas as pd
import requests
import json
from hygdra_forecasting.utils.sequence import create_combine_sequences, create_sequences_inference, create_sequences

SEQUENCE_LEN = 36
TIME_DELTA_LABEL = 12
TICKERS = ["AMD", "INTC", "SQ", "BA", "PFE", "PYPL", "COST", "SBUX", "DIS", "NFLX", 'GOOG', "NVDA", "JNJ", "META", "BRK-B", "GOOGL", "AAPL", "MSFT", "AMZN", "BTC-EUR", "ETH-EUR", "CRO-EUR", "AMZN", "BTC-USD", "ETH-USD", "CRO-USD", "INJ-USD", "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD"]
TICKERS_ETF = ["^GSPC", "^FCHI", "^IXIC","EBIT.TO", "BTC-USD"]

# indicateur:list[str]=TICKERS_ETF
def dataframe_to_graph_dataset(df:pd.DataFrame, indic_data:pd.DataFrame, labels:pd.DataFrame, tickers:list[str]=TICKERS):
    """
    transforme un Dataset de stock OHVL en serie de sequence temporel

    Args:
        df (pd.DataFrame): Série temporelle des prix de clôture.
        label (pd.DataFrame): ground truth
        tickers (List[str]): list des courts a predire
    Returns:
        Tuple[np.array, np.array] : sequences, label
    """
    indics_ticker = []
    for indics_col in indic_data.columns:
        indic_tick = indics_col.split('_')[0]
        if indic_tick not in indics_ticker:
            indics_ticker.append(indic_tick)

    arr = []
    for indic_tick in indics_ticker:
        # Extract close and volume data for the ticker
        close = indic_data[indic_tick+'_close'].values
        width = indic_data[indic_tick+'_width'].values
        rsi = indic_data[indic_tick+'_rsi'].values
        roc = indic_data[indic_tick+'_roc'].values
        volume = indic_data[indic_tick+'_volume'].values
        diff = indic_data[indic_tick+'_diff'].values
        pct_change = indic_data[indic_tick+'_percent_change_close'].values

        # Combine close and volume data
        ticker_data = np.column_stack((close,
                                    width,
                                    rsi,
                                    roc,
                                    volume,
                                    diff,
                                    pct_change))
        arr.append(ticker_data)

    sequences_dict = {}
    sequence_labels = {}
    for ticker in tickers:
        arr2 = arr.copy()
        # Extract close and volume data for the ticker
        close = df[ticker+'_close'].values
        width = df[ticker+'_width'].values
        rsi = df[ticker+'_rsi'].values
        roc = df[ticker+'_roc'].values
        volume = df[ticker+'_volume'].values
        diff = df[ticker+'_diff'].values
        pct_change = df[ticker+'_percent_change_close'].values

        # Combine close and volume data
        ticker_data = np.column_stack((close,
                                    width,
                                    rsi,
                                    roc,
                                    volume,
                                    diff,
                                    pct_change))
        arr2.append(ticker_data)

        # Generate sequences
        attribute = ticker+"_close"
        ticker_sequences, lab = create_combine_sequences(arr2,
                                                labels[attribute].values[SEQUENCE_LEN-1:])

        sequences_dict[ticker] = ticker_sequences
        sequence_labels[ticker] = lab
    
    # Combine data and labels from all tickers
    all_sequences = []
    all_labels = []

    for ticker in tickers:
        all_sequences.extend(sequences_dict[ticker])
        all_labels.extend(sequence_labels[ticker])

    # Convert to numpy arrays
    all_sequences = np.array(all_sequences)
    all_labels = np.array(all_labels)

    # split in another func ?
    return all_sequences, all_labels

def dataframe_to_dataset(df:pd.DataFrame, labels:pd.DataFrame, tickers:list[str]=TICKERS):
    """
    transforme un Dataset de stock OHVL en serie de sequence temporel

    Args:
        df (pd.DataFrame): Série temporelle des prix de clôture.
        label (pd.DataFrame): ground truth
        tickers (List[str]): list des courts a predire
    Returns:
        Tuple[np.array, np.array] : sequences, label
    """
    sequences_dict = {}
    sequence_labels = {}
    print("input shape", df.shape)
    for ticker in tickers:
        try : 
            # Extract close and volume data for the ticker
            close = df[ticker+'_close'].values
            width = df[ticker+'_width'].values
            rsi = df[ticker+'_rsi'].values
            roc = df[ticker+'_roc'].values
            volume = df[ticker+'_volume'].values
            diff = df[ticker+'_diff'].values
            pct_change = df[ticker+'_percent_change_close'].values

            # Combine close and volume data
            ticker_data = np.column_stack((close,
                                        width,
                                        rsi,
                                        roc,
                                        volume,
                                        diff,
                                        pct_change))

            # Generate sequences
            attribute = ticker+"_close"
            ticker_sequences, lab = create_sequences(ticker_data,
                                                    labels[attribute].values[SEQUENCE_LEN-1:])
            sequences_dict[ticker] = ticker_sequences
            sequence_labels[ticker] = lab
            
        except  :
            print("error at ", ticker)
            continue
    
    # Combine data and labels from all tickers
    all_sequences = []
    all_labels = []

    for ticker in sequences_dict.keys():
        if sequences_dict[ticker].shape[0] > 0 :
            print('loaded, shape', ticker, sequences_dict[ticker].shape)
            all_sequences.extend(sequences_dict[ticker])
            all_labels.extend(sequence_labels[ticker])

    # Convert to numpy arrays
    all_sequences = np.array(all_sequences)
    all_labels = np.array(all_labels)
    print("training shape ", all_sequences.shape)
    # split in another func ?
    return all_sequences, all_labels

def dataframe_to_dataset_inference(df:pd.DataFrame, tickers:list[str]=TICKERS):
    """
    transforme un Dataset de stock OHVL en serie de sequence temporel

    Args:
        df (pd.DataFrame): Série temporelle des prix de clôture.
        tickers (List[str]): list des courts a predire
    Returns:
        Tuple[np.array, np.array] : sequences, label
    """
    sequences_dict = {}
    for ticker in tickers:
        try:
            # Extract close and volume data for the ticker
            close = df[ticker+'_close'].values
            width = df[ticker+'_width'].values
            rsi = df[ticker+'_rsi'].values
            roc = df[ticker+'_roc'].values
            volume = df[ticker+'_volume'].values
            diff = df[ticker+'_diff'].values
            pct_change = df[ticker+'_percent_change_close'].values

            # Combine close and volume data
            ticker_data = np.column_stack((close,
                                        width,
                                        rsi,
                                        roc,
                                        volume,
                                        diff,
                                        pct_change))

            # Generate sequences
            ticker_sequences = create_sequences_inference(ticker_data)

            sequences_dict[ticker] = ticker_sequences
        except :
            print("ticker got seq issues :", ticker)

    # split in another func ?
    return sequences_dict

def dict_to_dataset(data_json: dict, labels_json: dict):
    """
    Transform a stock OHLCV dataset stored in JSON-like dictionary format into time series sequences.

    Args:
        data_json (dict): Dictionary containing stock OHLCV data.
        labels_json (dict): Ground truth labels.
        tickers (list[str]): List of tickers to process.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Sequences and labels.
    """
    sequences_dict = {}
    sequence_labels = {}
    print("Input data keys:", data_json.keys())
    print("label :", labels_json.keys())
    for ticker in data_json.keys():
        # Extract OHLCV features from JSON dictionary
        ticker_data_json = data_json.get(ticker, {})
        close = np.array(ticker_data_json.get("close", []), dtype=np.float64)
        width = np.array(ticker_data_json.get("width", []), dtype=np.float64)
        rsi = np.array(ticker_data_json.get("rsi", []), dtype=np.float64)
        roc = np.array(ticker_data_json.get("roc", []), dtype=np.float64)
        volume = np.array(ticker_data_json.get("volume", []), dtype=np.float64)
        diff = np.array(ticker_data_json.get("diff", []), dtype=np.float64)
        pct_change = np.array(ticker_data_json.get("percent_change_close", []), dtype=np.float64)
            
        # Combine features into a single NumPy array
        ticker_data = np.column_stack((close, width, rsi, roc, volume, diff, pct_change))
            
        # Generate sequences
        ticker_sequences, lab = create_sequences(ticker_data, np.array(labels_json.get(ticker))[SEQUENCE_LEN-1:])
        sequences_dict[ticker] = ticker_sequences
        sequence_labels[ticker] = lab
    
    # Combine data and labels from all tickers
    all_sequences = []
    all_labels = []
    
    for ticker, sequences in sequences_dict.items():
        if sequences.shape[0] > 0:
            print('Loaded, shape', ticker, sequences.shape)
            all_sequences.extend(sequences)
            all_labels.extend(sequence_labels[ticker])
    
    # Convert to numpy arrays
    all_sequences = np.array(all_sequences)
    all_labels = np.array(all_labels)
    print("Training shape", all_sequences.shape)
    
    return all_sequences, all_labels

def json_to_dataset_inference(data_json: dict, tickers: list[str] = TICKERS):
    """
    Transform a stock OHLCV dataset stored in JSON-like dictionary format into time series sequences for inference.

    Args:
        data_json (dict): Dictionary containing stock OHLCV data.
        tickers (list[str]): List of tickers to process.
    Returns:
        dict: Dictionary of sequences for each ticker.
    """
    sequences_dict = {}
    print("Input data keys:", data_json.keys())
    
    for ticker in tickers:
        try:
            # Extract OHLCV features from JSON dictionary
            ticker_data_json = data_json.get(ticker, {})
            close = np.array(ticker_data_json.get("close", []), dtype=np.float64)
            width = np.array(ticker_data_json.get("width", []), dtype=np.float64)
            rsi = np.array(ticker_data_json.get("rsi", []), dtype=np.float64)
            roc = np.array(ticker_data_json.get("roc", []), dtype=np.float64)
            volume = np.array(ticker_data_json.get("volume", []), dtype=np.float64)
            diff = np.array(ticker_data_json.get("diff", []), dtype=np.float64)
            pct_change = np.array(ticker_data_json.get("percent_change_close", []), dtype=np.float64)
            
            # Combine features into a single NumPy array
            ticker_data = np.column_stack((close, width, rsi, roc, volume, diff, pct_change))
            
            # Generate sequences
            ticker_sequences = create_sequences_inference(ticker_data)
            sequences_dict[ticker] = ticker_sequences
        
        except Exception as e:
            print(f"Ticker got sequence issues: {ticker}, Error: {e}")
            continue
    
    return sequences_dict
