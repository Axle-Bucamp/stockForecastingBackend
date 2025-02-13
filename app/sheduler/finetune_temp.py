from typing import List
from hygdra_forecasting.model.build import ConvCausalLTSM
from datamodel.ticker_cluster import TKGroup
from torch import save, load
from torch.utils.data import DataLoader
from torch import cuda, device
from hygdra_forecasting.dataloader.dataloader import StockDataset
from hygdra_forecasting.utils.learning_rate_sheduler import CosineWarmup
from hygdra_forecasting.model.train import train_model, setup_seed
from hygdra_forecasting.model.eval import validate
import torch.nn as nn

if cuda.is_available():
        device = device('cuda:0')
        print('Running on the GPU')
else:
    device = device('cpu')
    print('Running on the CPU')

# lineare sheduler
def tuning_sheduler(epoch:int):
    return 0.03 / (epoch + 1)

# for one group
def finetune_one(tickers:List[str], path:str):
    dataset = StockDataset(ticker=tickers)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=1)

    input_sample, _ = dataset.__getitem__(0)
    model = ConvCausalLTSM(input_shape=input_sample.shape)
    model = train_model(model, dataloader, epochs=10, learning_rate=0.01, save_epoch=False, lrfn=CosineWarmup(0.01, 10).lrfn, checkpoint_file=load('weight/best_model.pth'))
    save(model.state_dict(), path)
    
# for all groups
def finetune_many():
    setup_seed(20)
    # default
    seq = 'days'
    for group in TKGroup.__members__.values():
        group_name, tickers = group.value
        print(tickers)
        finetune_one(tickers, f'weight/{seq}/{group_name}.pth')


    
if __name__ == "__main__" :
    finetune_many()
    