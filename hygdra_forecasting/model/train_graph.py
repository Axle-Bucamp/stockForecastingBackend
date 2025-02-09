if __name__ == '__main__':
    # Example Usage:
    # Assume you already have the StockDataset and DataLoader set up as shown earlier
    from hygdra_forecasting.model.build import GraphforecastPred
    from hygdra_forecasting.dataloader.GraphDataloader import StockGraphDataset
    from hygdra_forecasting.model.train import train_model, setup_seed
    from torch.utils.data import DataLoader
    from hygdra_forecasting.utils.learning_rate_sheduler import CosineWarmup
    from torch import cuda, device


    if cuda.is_available():
        device = device('cuda:0')
        print('Running on the GPU')
    else:
        device = device('cpu')
        print('Running on the CPU')

    # work on model ? redo double chanel one conv causal the other as validator
    # liquid net / graph like llm
    tickers= ["DEFI", "PANW", "MRVL", "NKLA", "AFRM", "NKE", "BILL", "EXPE", 'LINK-USD', "TTWO", "NET", 'ICP-USD', 'FET-USD', 'FIL-USD', 'THETA-USD','AVAX-USD', 'HBAR-USD', 'UNI-USD', 'STX-USD', 'OM-USD', 'FTM-USD', "INJ-USD", "INTC", "SQ", "XOM", "COST", "BP", "BAC", "JPM", "GS", "CVX", "BA", "PFE", "PYPL", "SBUX", "DIS", "NFLX", 'GOOG', "NVDA", "JNJ", "META", "GOOGL", "AAPL", "MSFT", "BTC-EUR", "CRO-EUR", "ETH-USD", "CRO-USD", "BTC-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD"]
    tickers_val = ["AMZN", "AMD", "ETH-EUR", "ELF", "UBER"]
    TICKERS_ETF = ["DEFI", "AMZN", "^FCHI", "^IXIC","EBIT.TO", "BTC-USD", "ETH-EUR", "IXC"] # 7 min

    # tran data
    dataset = StockGraphDataset(ticker=tickers, indics=TICKERS_ETF)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=1)

    # val data
    dataset_val = StockGraphDataset(ticker=tickers_val, indics=TICKERS_ETF)
    dataloader_val = DataLoader(dataset_val, batch_size=128, shuffle=True, num_workers=1)

    # Initialize your model
    input_sample, _ = dataset.__getitem__(0)
    setup_seed(20)
    model = GraphforecastPred(input_shape=input_sample.shape)  # Modify this according to your dataset
    # dataloader_val
    model = train_model(model, dataloader, val_dataloader=dataloader_val, epochs=200, learning_rate=0.1, lrfn=CosineWarmup(0.1, 200).lrfn)

