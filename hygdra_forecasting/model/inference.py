from pandas import DataFrame
from hygdra_forecasting.utils.preprocessing import ohlv_to_dataframe_inference, dataframe_to_dataset_inference
from torch import tensor, no_grad, load

if __name__ == '__main__':
    # Example Usage:
    # Assume you already have the StockDataset and DataLoader set up as shown earlier
    from hygdra_forecasting.model.build import ConvCausalLTSM
    from torch import cuda, device

    if cuda.is_available():
        device = device('cuda:0')
        print('Running on the GPU')
    else:
        device = device('cpu')
        print('Running on the CPU')
    
    tickers= ["AMD", "INTC", "SQ", "BA", "PFE", "PYPL", "COST", "SBUX", "DIS", "NFLX", 'GOOG', "NVDA", "JNJ", "META", "GOOGL", "AAPL", "MSFT", "BTC-EUR", "ETH-EUR", "CRO-EUR", "AMZN", "ETH-USD", "CRO-USD", "INJ-USD", "BTC-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD"]
    
    df, dict_unorm = ohlv_to_dataframe_inference(tickers)
    sequences_dict = dataframe_to_dataset_inference(df, tickers)
    model = ConvCausalLTSM(input_shape=sequences_dict[tickers[0]][0].shape)  # Modify this according to your dataset
    model.load_state_dict(load('weight/basemodel.pt', weights_only=True))
    model.eval()

    df_result = DataFrame()
    # corriger iloc -100
    df_result.index = df[tickers[0] + "_close"].iloc[-100:].index
    for ticker in tickers:
        sequence = tensor(sequences_dict[ticker]).float()  # Convert to tensor
        
        with no_grad():
            predictions = model(sequence)

        predictions = predictions.squeeze().numpy() 
        print(dict_unorm[ticker][1], dict_unorm[ticker][0])
        df_result[ticker + "_pred"] = predictions.reshape(-1)[-100:] * dict_unorm[ticker][1] + dict_unorm[ticker][0]
        df_result[ticker + "_close"] = df[ticker + '_close'].iloc[-100:] * dict_unorm[ticker][1] + dict_unorm[ticker][0]

    print(df_result)
    