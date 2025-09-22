#TODO
if __name__ == '__main__':
    # Example Usage:
    # Assume you already have the StockDataset and DataLoader set up as shown earlier
    from hygdra_forecasting.model.train import setup_seed, train_model
    from hygdra_forecasting.utils.learning_rate_sheduler import CosineWarmup
    from hygdra_forecasting.model.build_liquid import ConvCausalLiquid
    from hygdra_forecasting.dataloader.dataloader import StockDataset
    from torch import device, cuda
    from torch.utils.data import DataLoader

    if cuda.is_available():
        device = device('cuda:0')
        print('Running on the GPU')
    else:
        device = device('cpu')
        print('Running on the CPU')

    # work on model ? redo double chanel one conv causal the other as validator
    # liquid net / graph like llm
    tickers= ["DEFI", "PANW", "MRVL", "NKLA", "AFRM", "EBIT.TO", "^FCHI", "NKE", "^GSPC", "^IXIC", "BILL", "EXPE", 'LINK-USD', "TTWO", "NET", 'ICP-USD', 'FET-USD', 'FIL-USD', 'THETA-USD','AVAX-USD', 'HBAR-USD', 'UNI-USD', 'STX-USD', 'OM-USD', 'FTM-USD', "INJ-USD", "INTC", "SQ", "XOM", "COST", "BP", "BAC", "JPM", "GS", "CVX", "BA", "PFE", "PYPL", "SBUX", "DIS", "NFLX", 'GOOG', "NVDA", "JNJ", "META", "GOOGL", "AAPL", "MSFT", "BTC-EUR", "CRO-EUR", "ETH-USD", "CRO-USD", "BTC-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD"]
    tickers_val = ["AMZN", "AMD", "ETH-EUR", "ELF", "UBER"]
    TICKERS_ETF = ["^GSPC", "^FCHI", "^IXIC","EBIT.TO", "BTC-USD"]

    # tran data
    dataset = StockDataset(ticker=tickers)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=1)

    # val data
    dataset_val = StockDataset(ticker=tickers_val)
    dataloader_val = DataLoader(dataset_val, batch_size=256, shuffle=True, num_workers=1)

    # Initialize your model
    input_sample, _ = dataset.__getitem__(0)
    setup_seed(20) # test liquid ? # check shaping and batch computation based on Dataset
    model = ConvCausalLiquid(input_shape=input_sample.shape)
    # LtsmAttentionforecastPred(input_shape=input_sample.shape)
    # ConvCausalLTSM(input_shape=input_sample.shape)
    model = train_model(model, dataloader, dataloader_val, epochs=100, learning_rate=0.01, lrfn=CosineWarmup(0.01, 100).lrfn)

