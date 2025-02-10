if __name__ == "__main__":
    from hygdra_forecasting.model.build import ConvCausalLTSM
    from hygdra_forecasting.dataloader.dataloader import StockDataset
    from hygdra_forecasting.model.eval import validate
    from torch import cuda, device
    from torch import load

    if cuda.is_available():
        device = device('cuda:0')
        print('Running on the GPU')
    else:
        device = device('cpu')
        print('Running on the CPU')

    criterion = nn.L1Loss() # You can adjust the penalty_factor
    tickers = ['BTC-USD']

    # val data
    dataset_val = StockDataset(ticker=tickers)
    dataloader_val = DataLoader(dataset_val, batch_size=256, shuffle=True, num_workers=1)

    # Initialize your model
    input_sample, _ = dataloader_val.__getitem__(0)

    model = ConvCausalLTSM(input_shape=input_sample.shape)  # Modify this according to your dataset
    model.load_state_dict(load('weight/epoch-380_loss-0.2699198153614998.pt', weights_only=True))

    print(validate(model, dataloader_val, criterion))