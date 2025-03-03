#TODO
# - enhance graph computation
# - dataset fron json
# - multimodality ?
# xformer

if __name__ == '__main__':
    # Example Usage:
    # Assume you already have the StockDataset and DataLoader set up as shown earlier
    from hygdra_forecasting.model.build_graph import GraphforecastPred, GraphTransformerforecastPred
    from hygdra_forecasting.dataloader.GraphDataloader import StockGraphDataset
    from hygdra_forecasting.model.train import train_model, setup_seed
    from torch.utils.data import DataLoader
    from hygdra_forecasting.utils.learning_rate_sheduler import CosineWarmup
    from torch import cuda, device, load, nn
    import numpy as np

    if cuda.is_available():
        device = device('cuda:0')
        print('Running on the GPU')
    else:
        device = device('cpu')
        print('Running on the CPU')

    # eliminer les mauvais stock ajuster learning rate + liquid net
    tickers= ["DEFI", "PANW", "MRVL", "NKLA", "AFRM", "NKE", "BILL", "EXPE", 'LINK-USD', "TTWO", "NET", 'THETA-USD','AVAX-USD', 'HBAR-USD', 'UNI-USD', 'STX-USD', "INTC", "SQ", "XOM", "COST", "BP", "BAC", "JPM", "GS", "CVX", "BA", "PFE", "PYPL", "SBUX", "DIS", "NFLX", 'GOOG', "NVDA", "JNJ", "META", "GOOGL", "AAPL", "MSFT", "BTC-EUR", "ETH-USD", "BTC-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD"]
    # tickers = ["BTC-USD"]
    tickers_val = ["AMZN", "AMD", "ETH-EUR", "ELF", "UBER"]
    TICKERS_ETF = ["^GSPC", "^FCHI", "^IXIC", "EBIT.TO", "BTC-USD"] # 7 min
    # TICKERS_ETF = ["BTC-USD"]

    # tran data
    dataset = StockGraphDataset(ticker=tickers, indics=TICKERS_ETF)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)

    # val data
    dataset_val = StockGraphDataset(ticker=tickers_val, indics=TICKERS_ETF)
    dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=True, num_workers=1)

    # temp (non distinct loss and balance) seq val on known stock
    lenval = len(dataset_val)
    indval = int(len(dataset) / 1.3)   # Select half the dataset

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

    setup_seed(20)
    #model = GraphforecastPred(input_shape=input_sample.shape)  # Modify this according to your dataset
    model = GraphTransformerforecastPred(input_shape=input_sample.shape)
    del input_sample
    # seems to loose context -> fix infrastructure , checkpoint_file=load('weight/best_model.pth')
    model = train_model(model, dataloader, val_dataloader=dataloader_val, epochs=100, learning_rate=0.01, lrfn=CosineWarmup(0.01, 100).lrfn, criterion=nn.L1Loss())