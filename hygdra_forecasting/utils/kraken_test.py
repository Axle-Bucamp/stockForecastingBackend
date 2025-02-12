import requests
import pandas as pd 
import json
import numpy as np
import yfinance as yf
from hygdra_forecasting.utils.preprocessing import calculate_bollinger_bands, calculate_roc, calculate_rsi


def get_kraken_data(ticker:str, interval:str) -> pd.DataFrame:
    pair = ticker.split("-")[0] + "USD"
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"

    payload = {}
    headers = {
    'Accept': 'application/json'
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    data = json.loads(response.text)
    # [int <time>, string <open>, string <high>, string <low>, string <close>, string <vwap>, string <volume>, int <count>]
    keys = list(data["result"].keys())
    data = pd.DataFrame(data["result"][keys[0]], columns=["Date", "Open", "High", "Low", "Close", "vwap", "Volume", "count"])
    data["Close"] = pd.to_numeric(data["Close"])
    data["High"] = pd.to_numeric(data["High"])
    data["Low"] = pd.to_numeric(data["Low"])
    data["Open"] = pd.to_numeric(data["Open"])
    data["Volume"] = pd.to_numeric(data["Volume"])
    data.drop(["count", "vwap"], axis=1, inplace=True)
    data.index = data["Date"]
    return data 


# generaliser interval
# Generalise ticker loader, three type : Daily, Hourly, Minutes
def ohlv_to_dataframe(tickers:list[str], period:str="2y", interval:str='1d'):
    kraken = False
    if interval in ["1", "5", "15", "30", "60", "240", "1440", "10080", "21600"]:
        kraken = True

    ticker_data_frames = []
    for ticker in tickers:
        # Download historical data for the ticker
        if kraken :
            data = get_kraken_data(ticker, interval)
        else :
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

if __name__ == "__main__":
    print(ohlv_to_dataframe(tickers=["BTC-USD", "XRP"], interval="1" ))