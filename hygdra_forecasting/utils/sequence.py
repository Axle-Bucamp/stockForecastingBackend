import numpy as np
import yfinance as yf
import pandas as pd
import requests
import json

SEQUENCE_LEN = 36
TIME_DELTA_LABEL = 12
TICKERS = ["AMD", "INTC", "SQ", "BA", "PFE", "PYPL", "COST", "SBUX", "DIS", "NFLX", 'GOOG', "NVDA", "JNJ", "META", "BRK-B", "GOOGL", "AAPL", "MSFT", "AMZN", "BTC-EUR", "ETH-EUR", "CRO-EUR", "AMZN", "BTC-USD", "ETH-USD", "CRO-USD", "INJ-USD", "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD"]
TICKERS_ETF = ["^GSPC", "^FCHI", "^IXIC","EBIT.TO", "BTC-USD"]

def create_combine_sequences(data:list[np.array], labels:pd.Series, sequence_length:int=SEQUENCE_LEN, time_delta:int=TIME_DELTA_LABEL) -> tuple[np.array, np.array]:
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
    data_size = len(data[-1])

    # Loop to create each sequence and its corresponding label
    for i in range(data_size - (sequence_length + 13)): # Ensure we have data for the label
        if i == 0:
          continue
        try :
            seq = []
            for stock in data:
                seq.append(stock[i:i + sequence_length])
            sequences.append(np.array(seq))  # The sequence of data
            lab.append( labels[i + time_delta] ) # The label and scaling factors
        except: # not homogenouis -> correct shape, detect wrong ticker
            continue

    return np.array(sequences), np.array(lab)

def create_sequences(data:np.array, labels:np.array, sequence_length:int=SEQUENCE_LEN, time_delta:int=TIME_DELTA_LABEL) -> tuple[np.array, np.array]:
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