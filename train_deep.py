if __name__ == '__main__':
    # Example Usage:
    # Assume you already have the StockDataset and DataLoader set up as shown earlier
    from hygdra_forecasting.model.train import setup_seed, train_model
    from hygdra_forecasting.utils.learning_rate_sheduler import CosineWarmup
    from hygdra_forecasting.model.build import ConvCausalLTSM, LtsmAttentionforecastPred
    from hygdra_forecasting.dataloader.dataloader import StockDataset
    from torch import device, cuda, load
    from torch.utils.data import DataLoader
    import json
    import numpy as np
    import random


    if cuda.is_available():
        device = device('cuda:0')
        print('Running on the GPU')
    else:
        device = device('cpu')
        print('Running on the CPU')

    # work on model ? redo double chanel one conv causal the other as validator
    
    # Load enabled assets from JSON
    with open("enabled_assets.json", "r") as file:
        enabled_assets = json.load(file)

    # Shuffle the list to ensure randomness
    random.shuffle(enabled_assets)

    #enabled_assets = enabled_assets # limit
    # Define split ratio (e.g., 80% train, 20% validation)
    split_ratio = 0.9
    split_index = int(len(enabled_assets) * split_ratio)

    # Split into training and validation sets
    tickers = enabled_assets[:split_index]
    tickers_val = enabled_assets[split_index:]
    
    # tran data
    # interval standard : ["1", "5", "15", "30", "60", "240", "1440", "10080", "21600"]
    # add for loop ? => fine tune on one interval + fine tune 1 epoch on focused one
    intervals = ["1", "5", "15", "30", "60", "240", "1440", "10080", "21600"]
    for interval in intervals:
        dataset = StockDataset(ticker=tickers, interval=interval) #, interval='1'
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=1)

        # val data
        dataset_val = StockDataset(ticker=tickers_val, interval=interval)
        dataloader_val = DataLoader(dataset_val, batch_size=256, shuffle=True, num_workers=1)

        # temp (non distinct loss and balance) seq val on known stock
        lenval = len(dataset_val)
        indval = int(len(dataset) / 1.1)  # Select half the dataset

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
        model = train_model(model, dataloader, dataloader_val, epochs=10, learning_rate=0.01, lrfn=CosineWarmup(0.01, 10).lrfn, checkpoint_file=load("weight/best_model.pth"))

