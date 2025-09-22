from hygdra_forecasting.model.build import ConvCausalLTSM
from datamodel.ticker_cluster import TKGroup
from torch import load, tensor, no_grad, cuda, device
from hygdra_forecasting.utils.ohlv import get_kraken_data_to_json
from hygdra_forecasting.utils.dataset import dict_to_dataset_inference
import numpy as np
import redis
from os import getenv
import json

class StockPredictor:
    def __init__(self, interval: str = 'days'):
        self.interval = interval
        self.redis_client = redis.Redis(
            host=getenv("REDIS_HOST", "localhost"), 
            port=int(getenv("REDIS_PORT", "6379")),
            db=0
        )
        self.device = device('cuda:0') if cuda.is_available() else device('cpu')
        print(f'Running on the {self.device}')
        self.model = ConvCausalLTSM((36, 7))
        self.interval_transform = {"days" : '1440', "minutes" : '1', "hours" : '60', "thrity" : "30"}

    
    def predict(self):
        for group in TKGroup.__members__.values():
            groups_name, tickers = group.value

            self.model.load_state_dict(load(f'weight/{self.interval}/best_model.pth')["model_state_dict"]) # {groups_name}.pth
            self.model.eval()
            
            df, dict_unorm, index_timestamp = get_kraken_data_to_json(tickers, interval=self.interval_transform[self.interval])
            sequences_dict = dict_to_dataset_inference(df)

            # 1) join all as before
            # 2) optimize json dict of dataframe
            for ticker in tickers:
                try :
                    ticker = ticker.split("-")[0] + "USD"
                    sequence = tensor(sequences_dict[ticker]).float()
                    
                    with no_grad():
                        predictions = self.model(sequence)
                    
                    date_array = np.array(index_timestamp[ticker], dtype='datetime64[s]')
                    df[ticker]["Date"] = date_array 
                    predictions = predictions.squeeze().numpy().reshape(-1)
                    
                    df[ticker]["forecasting"] = predictions * dict_unorm[ticker]["close"]["std"] + dict_unorm[ticker]["close"]["mean"]
                    df[ticker]["close"] = df[ticker]["close"] * dict_unorm[ticker]["close"]["std"] + dict_unorm[ticker]["close"]["mean"]

                    # prediction interval # (double check)
                    offset = None
                    if self.interval == "minutes":
                        offset = np.timedelta64(14, "m")
                    elif self.interval == "thirty":
                        offset= np.timedelta64(14 * 30, "m")
                    elif self.interval == "hours":
                        offset = np.timedelta64(14, "h")
                    else :
                        offset = np.timedelta64(14, "D")

                    df[ticker]["pred_date"] = df[ticker]["Date"] + offset
                    float_keys = ['close', 'forecasting', 'low', 'high', 'open', 'volume', 'upper', 'lower', 'width', 'rsi', 'roc', 'diff', 'percent_change_close']
                    for key in float_keys:
                        df[ticker][key] = df[ticker][key].astype(float).tolist()

                    date_keys = ['Date', 'pred_date']
                    for key in date_keys:
                        df[ticker][key] = df[ticker][key].astype(str).tolist()

                    redis_key = f"{ticker}_{self.interval}"

                except Exception as e:
                    print(f"error predicting stock : {e} on stock {ticker}")

                # register data
                try:
                    df_json = json.dumps(df[ticker])
                    self.redis_client.set(redis_key, df_json)
                    print(f"Saved predictions for group '{groups_name}' to Redis with key '{redis_key}'")
                except Exception as e:
                    print(f"Error saving predictions for group '{groups_name}' to Redis: {e}")

if __name__ == "__main__":
    predictor = StockPredictor(interval='days')
    predictor.predict()
