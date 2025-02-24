import numpy as np
import pandas as pd

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


def dict_calculate_bollinger_bands(data: np.ndarray, window: int = 10, num_of_std: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands for a time series.
    
    Args:
        data (np.ndarray): Time series closing prices.
        window (int, optional): Moving average window (default: 10).
        num_of_std (int, optional): Number of standard deviations (default: 2).
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Upper and lower Bollinger Bands.
    """
    rolling_mean = np.convolve(data, np.ones(window)/window, mode='valid')
    rolling_std = np.std([data[i:i+window] for i in range(len(data)-window+1)], axis=1)
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return upper_band, lower_band

def dict_calculate_rsi(data: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Calculate the Relative Strength Index (RSI).
    
    Args:
        data (np.ndarray): Time series closing prices.
        window (int, optional): RSI calculation period (default: 10).
    
    Returns:
        np.ndarray: RSI values.
    """
    delta = np.diff(data, prepend=data[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gain, np.ones(window)/window, mode='valid')
    avg_loss = np.convolve(loss, np.ones(window)/window, mode='valid')
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def dict_calculate_roc(data: np.ndarray, periods: int = 10) -> np.ndarray:
    """
    Calculate Rate of Change.
    
    Args:
        data (np.ndarray): Time series closing prices.
        periods (int, optional): Period for ROC calculation (default: 10).
    
    Returns:
        np.ndarray: ROC values.
    """
    roc = ((data[periods:] - data[:-periods]) / data[:-periods]) * 100
    return np.concatenate((np.full(periods, np.nan), roc))

def dict_calculate_diff_and_pct_change(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the first-order difference and percentage change of a time series.
    
    Args:
        data (np.ndarray): Time series closing prices.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Difference and percentage change.
    """
    diff = np.diff(data, prepend=data[0])
    percent_change_close = np.concatenate(([0], (np.diff(data) / data[:-1]) * 100))
    return diff, percent_change_close
