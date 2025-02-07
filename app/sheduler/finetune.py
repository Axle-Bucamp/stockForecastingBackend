from typing import List
from hygdra_forecasting.model.build import ConvCausalLTSM
from datamodel.ticker_cluster import TKGroup
from torch import save, load
from torch.utils.data import DataLoader
from torch import cuda, device
from hygdra_forecasting.dataloader.dataloader import StockDataset
from hygdra_forecasting.utils.learning_rate_sheduler import CosineWarmup
from hygdra_forecasting.model.train import train_model

if cuda.is_available():
        device = device('cuda:0')
        print('Running on the GPU')
else:
    device = device('cpu')
    print('Running on the CPU')

# for one group
def finetune_one(tickers:List[str], path:str):
    dataset = StockDataset(ticker=tickers)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=1)

    input_sample, _ = dataset.__getitem__(0)
    model = ConvCausalLTSM(input_shape=input_sample.shape)

    model.load_state_dict(load('weight/best_model.pt', weights_only=True))  
    model = train_model(model, dataloader, epochs=10, learning_rate=0.03, save_epoch=False, lrfn=CosineWarmup(0.03, 10).lrfn)
    save(model.state_dict(), path)
    
# for all groups
def finetune_many():
    # default
    seq = 'days'
    for group in TKGroup.__members__.values():
        finetune_one(group.value[1], f'weight/{seq}/{group.value[0]}.pt')
    
if __name__ == "__main__" :
    finetune_many()
    