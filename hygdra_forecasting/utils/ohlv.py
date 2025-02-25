import pandas as pd 
import numpy as np
import json
import requests
import yfinance as yf
from hygdra_forecasting.utils.indicator import calculate_bollinger_bands, calculate_roc, calculate_rsi 
from hygdra_forecasting.utils.indicator import dict_calculate_bollinger_bands, dict_calculate_roc, dict_calculate_rsi, dict_calculate_diff_and_pct_change 


def get_kraken_data_to_dataframe(ticker:str, interval:str) -> pd.DataFrame:
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
    data.index = pd.to_datetime(data["Date"], unit='s').round('T')
    data.drop(["count", "vwap", "Date"], axis=1, inplace=True)
    print("======>", ticker)
    return data 

def get_kraken_data_to_json(tickers:list[str], interval:str) -> tuple[dict, dict, dict]:
    pairs = [ticker.split("-")[0] + "USD" for ticker in tickers]
    data_dict = {}
    for pair in pairs:
        url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"

        payload = {}
        headers = {
        'Accept': 'application/json'
        }

        response = requests.request("GET", url, headers=headers, data=payload)

        data = json.loads(response.text)

        for key, values in data["result"].items():
            if key != "last" :
                data_dict[pair] = values

    # [int <time>, string <open>, string <high>, string <low>, string <close>, string <vwap>, string <volume>, int <count>]
    processed_data, unorm_dict, time_index = kraken_preprocessing(data_dict)
    return processed_data, unorm_dict, time_index

def kraken_preprocessing(data: dict) -> tuple[dict, dict, dict]:
    """
    Processes Kraken OHLCV data stored in a dictionary format.
    Extracts close, low, high, open, and volume values for further analysis.
    """
    processed_data = {}
    unnorm_dict = {}

    # [int <time>, string <open>, string <high>, string <low>, string <close>, string <vwap>, string <volume>, int <count>]
    # data.pop("last")
    time_index = {}
    for key, values in data.items():
        if key != "last" :
            values = pd.DataFrame(values, columns=["Date", "Open", "High", "Low", "Close", "vwap", "Volume", "count"])
            ## repartir sur un mode dataframe ?

            processed_data[key] = {
                "close": np.array(values["Close"], dtype=np.float64),
                "low": np.array(values["Low"], dtype=np.float64),
                "high": np.array(values["High"], dtype=np.float64),
                "open": np.array(values["Open"], dtype=np.float64),
                "volume": np.array(values["Volume"], dtype=np.float64),
            }
            time_index[key] = np.array(values["Date"])
            
            processed_data[key]["upper"], processed_data[key]["lower"] = dict_calculate_bollinger_bands(processed_data[key]["close"], window=14, num_of_std=2)
            processed_data[key]["width"] = processed_data[key]["upper"] - processed_data[key]["lower"]

            processed_data[key]["rsi"] = dict_calculate_rsi(processed_data[key]["close"], window=14)
            processed_data[key]["roc"] = dict_calculate_roc(processed_data[key]["close"], periods=14)
            processed_data[key]["diff"], processed_data[key]["percent_change_close"] = dict_calculate_diff_and_pct_change(processed_data[key]["close"])

            # corriger la shape 
            time_index[key] = time_index[key][14:]
            for indic in processed_data[key].keys():
                if indic not in ["upper", "lower", "width", "rsi", "roc"]:
                    processed_data[key][indic] = processed_data[key][indic][14:]
                elif indic in ["upper", "lower", "width", "rsi"]:
                    processed_data[key][indic] = processed_data[key][indic][1:]

                MEAN = np.mean(processed_data[key][indic])
                STD = np.std(processed_data[key][indic])
                if STD == 0 :
                    STD = 1
                processed_data[key][indic] = (processed_data[key][indic] - MEAN) / STD
                # print(STD, MEAN, indic, key)
                if key not in unnorm_dict.keys():
                    unnorm_dict[key] = {indic : {"mean" : MEAN, "std" : STD}}
                else :
                    unnorm_dict[key][indic] = {"mean" : MEAN, "std" : STD}

    return processed_data, unnorm_dict, time_index

def preprocessing_training_mode(processed_data:dict):
    label = {}
    for key, values in processed_data.items():
        label[key] = {}
        shifted_series = np.array(values["close"][1:], dtype=np.float64)  # Shift data by one step
        label[key] = shifted_series
        
        # Ensure consistent data pairs by trimming the last entry
        for indic in processed_data[key].keys():
            processed_data[key][indic] = processed_data[key][indic][:-1]

    return processed_data, label 


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
    kraken = False
    if interval in ["1", "5", "15", "30", "60", "240", "1440", "10080", "21600"]:
        kraken = True

    ticker_data_frames = []
    for ticker in tickers:
        try : 
            # Download historical data for the ticker
            if kraken :
                data = get_kraken_data_to_dataframe(ticker, interval)
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
            STD.replace(0, 1, inplace=True)
            ticker_df = (ticker_df - MEAN) / STD

            ticker_df.replace([np.inf, -np.inf], 0, inplace=True)
            ticker_df.dropna(inplace=True)
            if ticker_df.shape[0] > 700: 
                # Normalize the training features
                ticker_data_frames.append(ticker_df)
        except :
            print("error at ", ticker)
            continue

    df = pd.concat(ticker_data_frames, axis=1)
    # to many stock leads to nan
    # df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True, axis=0)
    print("total shape", df.shape)

    # Shift the dataframe up by one to align current features with the next step's outcomes
    labels = df.shift(-1) # nb days shift, check to return a full sequence to predict N

    # Remove the last row from both the features and labels to maintain consistent data pairs
    df = df.iloc[:-1]
    labels = labels.iloc[:-1]

    return df, labels