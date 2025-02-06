from hygdra_forecasting.model.build import ConvCausalLTSM
from datamodel.ticker_cluster import TKGroup
from torch import load
from torch.utils.data import DataLoader
from torch import cuda, device
from hygdra_forecasting.dataloader.dataloader import StockDataset
from hygdra_forecasting.model.train import train_model
from hygdra_forecasting.utils.preprocessing import ohlv_to_dataframe_inference, dataframe_to_dataset_inference
from hygdra_forecasting.utils.learning_rate_sheduler import CosineWarmup
from torch import tensor, float32, no_grad
from pandas import DataFrame, DateOffset

if cuda.is_available():
    device = device('cuda:0')
    print('Running on the GPU')
else:
    device = device('cpu')
    print('Running on the CPU')

model = ConvCausalLTSM((36, 7)) 

def predict_daily():
    # get ticker groups
    groups_name = ''
    for group in TKGroup.__members__.values():
        groups_name, tickers = group.value

        # load group weight
        global model
        model.load_state_dict(load(f'weight/days/{groups_name}.pt', weights_only=True))
        model.eval()
        
        # get inference
        df, dict_unorm = ohlv_to_dataframe_inference(tickers)
        sequences_dict = dataframe_to_dataset_inference(df, tickers)

        df_result = DataFrame()
        df_result.index = df[tickers[0] + "_close"].iloc[-100:].index
        for ticker in tickers:
            sequence = tensor(sequences_dict[ticker]).float()  # Convert to tensor
            
            with no_grad():
                predictions = model(sequence)

            predictions = predictions.squeeze().numpy() 
            df_result[ticker + "_pred"] = predictions.reshape(-1)[-100:] * dict_unorm[ticker][1] + dict_unorm[ticker][0]
            df_result[ticker + "_close"] = df[ticker + '_close'].iloc[-100:] * dict_unorm[ticker][1] + dict_unorm[ticker][0]
        
        df_result["pred_date"] =  df_result.index + DateOffset(days=14)
     
        # Return the prediction (for simplicity, we return raw output here)
        df_result.to_csv(f'data/{groups_name}_days.csv')

if __name__ == "__main__":
    predict_daily()