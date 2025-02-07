import numpy as np
import yfinance as yf
import pandas as pd


SEQUENCE_LEN = 36
TIME_DELTA_LABEL = 12
TICKERS = ["AMD", "INTC", "SQ", "BA", "PFE", "PYPL", "COST", "SBUX", "DIS", "NFLX", 'GOOG', "NVDA", "JNJ", "META", "BRK-B", "GOOGL", "AAPL", "MSFT", "AMZN", "BTC-EUR", "ETH-EUR", "CRO-EUR", "AMZN", "BTC-USD", "ETH-USD", "CRO-USD", "INJ-USD", "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD"]

def calculate_bollinger_bands(data:pd.DataFrame, window:int=10, num_of_std:int=2) -> tuple[pd.Series, pd.Series]:
    """
    Calcule les bandes de Bollinger pour une série temporelle.

    Args:
        data (pd.DataFrame): Série temporelle des prix de clôture.
        window (int, optionnel): Fenêtre de calcul de la moyenne mobile (par défaut 10).
        num_of_std (int, optionnel): Nombre d'écarts types pour les bandes (par défaut 2).

    Returns:
        tuple[pd.Series, pd.Series]: Bandes supérieure et inférieure.
    """
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return upper_band, lower_band # add more moving average ?

def calculate_rsi(data: pd.DataFrame, window:int=10)-> pd.Series:
    """
    Calcule l'indice de force relative (RSI).

    Args:
        data (pd.DataFrame): Série temporelle des prix de clôture.
        window (int, optionnel): Période pour le calcul du RSI (par défaut 10).

    Returns:
        pd.Series: Valeurs du RSI.
    """
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_roc(data: pd.DataFrame, periods:int=10) -> pd.Series:
    """
    Calculate Rate of Change.

    Args:
        data (pd.DataFrame): Série temporelle des prix de clôture.
        period (int, optionnel): Période pour le calcul du RSI (par défaut 10).

    Returns:
        pd.Series: Valeurs du ROC.
    """
    roc = ((data - data.shift(periods)) / data.shift(periods)) * 100
    return roc

# vérifier
def create_sequences(data:np.array, labels:pd.Series, sequence_length:int=SEQUENCE_LEN, time_delta:int=TIME_DELTA_LABEL) -> tuple[np.array, np.array]:
    """
    Transforme une serie de cours en sequence ND

    Args:
        data (np.array): Série temporelle des prix de clôture.
        label (pd.Series): valeur reel d evaluation
        sequence_length (int): taille de la sequence temporel en entree
        time_delta (int): ecart temporel entre les entree et la sortie a predire

    Returns:
        tuple[np.array, np.array] : training sequences, labels
    """
    sequences = []
    lab = []
    data_size = len(data)

    # Loop to create each sequence and its corresponding label
    for i in range(data_size - (sequence_length + 13)): # Ensure we have data for the label
        if i == 0:
          continue
        sequences.append(data[i:i + sequence_length])  # The sequence of data
        lab.append( labels[i + time_delta] ) # The label and scaling factors

    return np.array(sequences), np.array(lab)

# Generalise ticker loader, three type : Daily, Hourly, Minutes
def ohlv_to_dataframe(tickers:list[str], period:str="2y", interval:str='1d'):
    """
    transforme une aquisition OHLV de court en DataFrame d analyse financiere
    elimine les valeurs sans label

    Args:
        tickers (List[str]): list des courts a predire
        period (str)=2y: period temporel d analyse (now - period)
        interval (str)=1d: interval temporel d analyse
    Returns:
        Tuple[pd.DataFrame, dict] : donnees d analyse, indice de denormalisation
    """
    ticker_data_frames = []
    for ticker in tickers:
        # Download historical data for the ticker
        data = yf.download(ticker, period=period, interval=interval) # checker pour faire un model jour heur minute

        # Calculate the daily percentage change
        close = data['Close']
        upper, lower = calculate_bollinger_bands(close, window=14, num_of_std=2)
        width = upper - lower
        rsi = calculate_rsi(close, window=14)
        roc = calculate_roc(close, periods=14)
        volume = data['Volume']
        diff = data['Close'].diff(1)
        percent_change_close = data['Close'].pct_change() * 100

        # Create a DataFrame for the current ticker and append it to the list
        ticker_df = pd.DataFrame({
            ticker+'_close': close.squeeze(),
            ticker+'_width': width.squeeze(),
            ticker+'_rsi': rsi.squeeze(),
            ticker+'_roc': roc.squeeze(),
            ticker+'_volume': volume.squeeze(),
            ticker+'_diff': diff.squeeze(),
            ticker+'_percent_change_close': percent_change_close.squeeze(),
        }, index=close.index)

        MEAN = ticker_df.mean()
        STD = ticker_df.std()

        # Normalize the training features
        ticker_df = (ticker_df - MEAN) / STD
        ticker_data_frames.append(ticker_df)

    df = pd.concat(ticker_data_frames, axis=1)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Shift the dataframe up by one to align current features with the next step's outcomes
    labels = df.shift(-1) # nb days shift, check to return a full sequence to predict N

    # Remove the last row from both the features and labels to maintain consistent data pairs
    df = df.iloc[:-1]
    labels = labels.iloc[:-1]

    return df, labels

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
    for ticker in tickers:

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

def create_sequences_inference(data, sequence_length=SEQUENCE_LEN):
    """
    Transforme une serie de cours en sequence ND pour l inference

    Args:
        data (np.array): Série temporelle des prix de clôture.
        sequence_length (int): taille de la sequence temporel en entree
    Returns:
        np.array : sequences
    """
    sequences = []
    data_size = len(data)

    # Loop to create each sequence and its corresponding label
    for i in range(data_size - sequence_length): # Ensure we have data for the label
        if i == 0:
          continue
        sequences.append(data[i:i + sequence_length])  # The sequence of dat

    return np.array(sequences)

# Generalise ticker loader, three type : Daily, Hourly, Minutes
def ohlv_to_dataframe_inference(tickers:list[str], period:str="2y", interval:str='1d'):
    """
    transforme une aquisition OHLV de court en DataFrame d analyse financiere

    Args:
        tickers (List[str]): list des courts a predire
        period (str)=2y: period temporel d analyse (now - period)
        interval (str)=1d: interval temporel d analyse
    Returns:
        Tuple[pd.DataFrame, dict] : donnees d analyse, indice de denormalisation
    """
    ticker_data_frames = []
    dict_unorm = {}
    for ticker in tickers:
        # Download historical data for the ticker
        data = yf.download(ticker, period=period, interval=interval) # checker pour faire un model jour heur minute

        # Calculate the daily percentage change
        close = data['Close']
        upper, lower = calculate_bollinger_bands(close, window=14, num_of_std=2)
        width = upper - lower
        rsi = calculate_rsi(close, window=14)
        roc = calculate_roc(close, periods=14)
        volume = data['Volume']
        diff = data['Close'].diff(1)
        percent_change_close = data['Close'].pct_change() * 100

        # Create a DataFrame for the current ticker and append it to the list
        ticker_df = pd.DataFrame({
            ticker+'_close': close.squeeze(),
            ticker+'_width': width.squeeze(),
            ticker+'_rsi': rsi.squeeze(),
            ticker+'_roc': roc.squeeze(),
            ticker+'_volume': volume.squeeze(),
            ticker+'_diff': diff.squeeze(),
            ticker+'_percent_change_close': percent_change_close.squeeze(),
        }, index=close.index)

        MEAN = ticker_df.mean()
        STD = ticker_df.std()

        # Normalize the training features
        dict_unorm[ticker] = (ticker_df[ticker+'_close'].mean(), ticker_df[ticker+'_close'].std())
        ticker_df = (ticker_df - MEAN) / STD
        ticker_data_frames.append(ticker_df)
        
    df = pd.concat(ticker_data_frames, axis=1)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Remove the last row from both the features and labels to maintain consistent data pairs
    df = df.iloc[:-1]

    return df, dict_unorm

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

    # split in another func ?
    return sequences_dict