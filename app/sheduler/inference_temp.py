from hygdra_forecasting.model.build import ConvCausalLTSM
from datamodel.ticker_cluster import TKGroup
from torch import load
from torch import cuda, device
from hygdra_forecasting.utils.preprocessing import ohlv_to_dataframe_inference, dataframe_to_dataset_inference
from torch import tensor, no_grad
from pandas import DataFrame, DateOffset
import redis
from os import getenv

redis_client = redis.Redis(
    host=getenv("REDIS_HOST", "localhost"), 
    port=int(getenv("REDIS_PORT", "6379")),  # Ensure port is an integer
    db=0
)

if cuda.is_available() and not device:
    device = device('cuda:0')
    print('Running on the GPU')
else:
    device = device('cpu')
    print('Running on the CPU')

model = ConvCausalLTSM((36, 7)) 

def predict_daily():
    """
    For each ticker group defined in TKGroup, load the corresponding model weights,
    generate predictions on the last 100 data points, and store the resulting DataFrame 
    as JSON in a Redis database.
    """
    # get ticker groups
    groups_name = ''
    for group in TKGroup.__members__.values():
        groups_name, tickers = group.value

        # load group weight
        global model
        global redis_client
        # f'weight/days/{groups_name}.pth'
        model.load_state_dict(load(f'weight/days/{groups_name}.pth'))
        model.eval()
        
        # get inference
        df, dict_unorm = ohlv_to_dataframe_inference(tickers)
        sequences_dict = dataframe_to_dataset_inference(df, tickers)

        df_result = DataFrame()
        df_result.index = df[tickers[0] + "_close"].iloc[-100:].index
        df_result["Date"] = df_result.index
        for ticker in tickers:
            sequence = tensor(sequences_dict[ticker]).float()  # Convert to tensor
            
            with no_grad():
                predictions = model(sequence)

            predictions = predictions.squeeze().numpy() 
            df_result[ticker + "_pred"] = predictions.reshape(-1)[-100:] * dict_unorm[ticker][1] + dict_unorm[ticker][0]
            df_result[ticker + "_close"] = df[ticker + '_close'].iloc[-100:] * dict_unorm[ticker][1] + dict_unorm[ticker][0]
        
            print(dict_unorm[ticker][1], dict_unorm[ticker][0])
        df_result["pred_date"] =  df_result.index + DateOffset(days=14)
     
        # Return the prediction (for simplicity, we return raw output here)
        df_result.to_csv(f'data/{groups_name}_days.csv')

        # Convert the result DataFrame to JSON
        json_data = df_result.to_json(orient="records", date_format="iso")
        redis_key = f"{groups_name}_days"
        
        # Save the JSON data into Redis
        try:
            redis_client.set(redis_key, json_data)
            print(f"Saved predictions for group '{groups_name}' to Redis with key '{redis_key}'")
        except Exception as e:
            print(f"Error saving predictions for group '{groups_name}' to Redis: {e}")
            



if __name__ == "__main__":
    predict_daily()