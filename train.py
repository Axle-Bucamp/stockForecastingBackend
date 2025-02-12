if __name__ == '__main__':
    # Example Usage:
    # Assume you already have the StockDataset and DataLoader set up as shown earlier
    from hygdra_forecasting.model.train import setup_seed, train_model
    from hygdra_forecasting.utils.learning_rate_sheduler import CosineWarmup
    from hygdra_forecasting.model.build import ConvCausalLTSM, LtsmAttentionforecastPred
    from hygdra_forecasting.dataloader.dataloader import StockDataset
    from torch import device, cuda, load
    from torch.utils.data import DataLoader
    import numpy as np

    if cuda.is_available():
        device = device('cuda:0')
        print('Running on the GPU')
    else:
        device = device('cpu')
        print('Running on the CPU')

    # work on model ? redo double chanel one conv causal the other as validator
    """
    # liquid net / graph like llm
    tickers= ["DEFI", "PANW", "MRVL", "NKLA", "AFRM", "EBIT.TO", "^FCHI", "NKE", "^GSPC", "^IXIC", "BILL", "EXPE", 'LINK-USD', "TTWO", "NET", 'ICP-USD', 'FET-USD', 'FIL-USD', 'THETA-USD','AVAX-USD', 'HBAR-USD', 'UNI-USD', 'STX-USD', 'OM-USD', 'FTM-USD', "INJ-USD", "INTC", "SQ", "XOM", "COST", "BP", "BAC", "JPM", "GS", "CVX", "BA", "PFE", "PYPL", "SBUX", "DIS", "NFLX", 'GOOG', "NVDA", "JNJ", "META", "GOOGL", "AAPL", "MSFT", "BTC-EUR", "CRO-EUR", "ETH-USD", "CRO-USD", "BTC-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD"]
    tickers_val = ["AMZN", "AMD", "ETH-EUR", "ELF", "UBER"]
    TICKERS_ETF = ["^GSPC", "^FCHI", "^IXIC","EBIT.TO", "BTC-USD"]
    """

    # live mode
    tickers= [ "CRO-EUR", "ETH-USD", "CRO-USD", "BTC-USD", "XRP-USD", "ADA-USD", "SOL-USD"]
    tickers_val = ['LINK-USD', 'ICP-USD', 'FET-USD', 'FIL-USD', "ETH-EUR"]

    # tran data
    dataset = StockDataset(ticker=tickers, interval='1')
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=1)

    # val data
    dataset_val = StockDataset(ticker=tickers_val)
    dataloader_val = DataLoader(dataset_val, batch_size=256, shuffle=True, num_workers=1)

    # temp (non distinct loss and balance) seq val on known stock
    lenval = len(dataset_val)
    indval = int(len(dataset) / 1.3)  # Select half the dataset

    # Ensure index bounds are valid 
    if indval > 0:
        # Select random indices without replacement
        random_indices = np.random.choice(len(dataset), indval, replace=False)
        
        # Move selected data to dataset_val
        dataset_val.data = np.concatenate((dataset_val.data, dataset.data[random_indices].copy()), axis=0)
        dataset_val.label = np.concatenate((dataset_val.label, dataset.label[random_indices].copy()), axis=0)

        # Remove selected indices from dataset
        mask = np.ones(len(dataset), dtype=bool)
        mask[random_indices] = False

        dataset.data = dataset.data[mask]
        dataset.label = dataset.label[mask]

    # Initialize your model
    input_sample, _ = dataset.__getitem__(0)

    # clear a bit of memory
    del mask
    del random_indices

    setup_seed(20) # test liquid ? # check shaping and batch computation based on Dataset
    model = ConvCausalLTSM(input_shape=input_sample.shape)
    del input_sample
    # LtsmAttentionforecastPred, ConvCausalLTSM
    model = train_model(model, dataloader, dataloader_val, epochs=100, learning_rate=0.01, lrfn=CosineWarmup(0.01, 100).lrfn, checkpoint_file=load("weight/standard/ConvCausalLTSM/80_weight.pth"))

