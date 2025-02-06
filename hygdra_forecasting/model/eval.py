import torch.nn as nn
from torch.utils.data import DataLoader
from torch import no_grad


def validate(model:nn.Module, dataloader:DataLoader, criterion):
    """
    validate the model based on given criterion

    arg :
        model (nn.Module): pytorch model
        dataloader (Dataloader): iterable dataloader class
        criterion (function): loss computation method
    return 
        float : loss rate on val data
    """
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with no_grad():  # No gradients needed for validation
        for batch_idx, (x, y) in enumerate(dataloader):
            y_pred = model(x)
            loss = criterion(y_pred, y)
            val_loss += loss.item()

    val_loss = val_loss / len(dataloader)
    return val_loss

if __name__ == "__main__":
    from hygdra_forecasting.model.build import ConvCausalLTSM
    from hygdra_forecasting.dataloader.dataloader import StockDataset
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