from hygdra_forecasting.model.build import ConvCausalLTSM
from datamodel.ticker_cluster import TKGroup
from torch import load, tensor, no_grad, cuda, device
from hygdra_forecasting.utils.preprocessing import ohlv_to_dataframe_inference, dataframe_to_dataset_inference
from pandas import DataFrame, DateOffset
import redis
from os import getenv

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
            
            df, dict_unorm = ohlv_to_dataframe_inference(tickers, interval=self.interval_transform[self.interval])
            sequences_dict = dataframe_to_dataset_inference(df, tickers)
            
            df_result = DataFrame()
            df_result.index = df[tickers[0] + "_close"].iloc[-100:].index
            df_result["Date"] = df_result.index
            
            for ticker in tickers:
                sequence = tensor(sequences_dict[ticker]).float()
                
                with no_grad():
                    predictions = self.model(sequence)
                
                predictions = predictions.squeeze().numpy()
                df_result[ticker + "_pred"] = predictions.reshape(-1)[-100:] * dict_unorm[ticker][1] + dict_unorm[ticker][0]
                df_result[ticker + "_close"] = df[ticker + '_close'].iloc[-100:] * dict_unorm[ticker][1] + dict_unorm[ticker][0]
                
            df_result["pred_date"] = df_result.index + DateOffset(days=14)
            df_result.to_csv(f'data/{groups_name}_{self.interval}.csv')
            
            json_data = df_result.to_json(orient="records", date_format="iso")
            redis_key = f"{groups_name}_{self.interval}"
            
            try:
                self.redis_client.set(redis_key, json_data)
                print(f"Saved predictions for group '{groups_name}' to Redis with key '{redis_key}'")
            except Exception as e:
                print(f"Error saving predictions for group '{groups_name}' to Redis: {e}")

if __name__ == "__main__":
    predictor = StockPredictor(interval='days')
    predictor.predict()
