import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch import save, load, manual_seed, device, cuda
from torch.backends import cudnn
from hygdra_forecasting.model.eval import validate
from hygdra_forecasting.utils.learning_rate_sheduler import CosineWarmup
import os

# Training Loop
def train_model(model:nn.Module, dataloader:DataLoader, val_dataloader:DataLoader=None, epochs:int=100, learning_rate:float=0.001, save_epoch=True, lrfn=CosineWarmup().lrfn, checkpoint_file=None):
    """
    Train a deep learning model using pytorch framework adapted for this case study
    save directory ./weight

    Args:
        model (nn.Module): current model instance
        dataloader (DataLoader): training dataloader
        val_dataloader (DataLoader) optional: validation dataloader
        epochs (int) optional: number of loop
        learning_rate (float) optional: starting learning rate applyed at each step
        save_epoch (bool) optional: save every n epochs 
        lrfn (callable) optional: sheduler for the learning rate, must take epoch:int as param

    Returns:
        nn.Module: train model
    """
    criterion = nn.MSELoss() # nn.L1Loss() # You can adjust the penalty_factor
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    scheduler = lr_scheduler.LambdaLR(optimizer, lrfn)
    best_val_loss = float('inf')

    if checkpoint_file :
        model.load_state_dict(checkpoint_file["model_state_dict"])
        optimizer.load_state_dict(checkpoint_file["optimizer_state_dict"])

    # Train the model
    model.train()  # Set model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()  # Clear gradients from previous step

            # Forward pass
            y_pred = model(x)

            # Calculate custom loss
            loss = criterion(y_pred, y)

            # Backward pass: Compute gradients
            loss.backward()

            # Optimize: Update model parameters
            optimizer.step()

            running_loss += loss.item()

        # Print the average loss for the current epoch
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(dataloader)}, lr: {optimizer.param_groups[0]['lr']}")
        scheduler.step()

        if val_dataloader :
            # Validation Phase
            val_loss = validate(model, val_dataloader, criterion)
            print(f"Validation Loss after Epoch [{epoch+1}/{epochs}]: {val_loss}")

            # Save the model with the best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_epoch:
                    save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, "weight/best_model.pth")
                    # save(model.state_dict(), 'weight/best_model.pth')

        if epoch % 40 == 0 and save_epoch:
            save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, f"weight/{epoch}_weight.pth")

    print("Training complete")
    return model

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    manual_seed(seed)
    cuda.manual_seed_all(seed)
    cudnn.deterministic = True

if __name__ == '__main__':
    # Example Usage:
    # Assume you already have the StockDataset and DataLoader set up as shown earlier
    from hygdra_forecasting.model.build import ConvCausalLTSM, GraphforecastPred
    from hygdra_forecasting.dataloader.dataloader import StockDataset


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
    setup_seed(20)
    model = GraphforecastPred(input_shape=input_sample.shape)  # Modify this according to your dataset
    model = train_model(model, dataloader, dataloader_val, epochs=100, learning_rate=0.01, lrfn=CosineWarmup(0.01, 100).lrfn)

