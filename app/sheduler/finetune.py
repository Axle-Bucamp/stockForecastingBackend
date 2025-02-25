from typing import List
from hygdra_forecasting.model.build import ConvCausalLTSM
from datamodel.ticker_cluster import TKGroup
from torch import save, load
from torch.utils.data import DataLoader
from torch import cuda, device
from hygdra_forecasting.dataloader.dataloader import StockDataset as standard
from hygdra_forecasting.dataloader.dataloader_kraken import StockDataset
from hygdra_forecasting.utils.learning_rate_sheduler import CosineWarmup
from hygdra_forecasting.model.train import train_model, setup_seed
import torch.nn as nn

class StockFineTuner:
    def __init__(self, interval: str = 'days', base_weight: str = 'weight/days/best_model.pth', epoch=100, learnig_rate=0.01):
        self.interval = interval
        self.base_weight = base_weight
        self.device = device('cuda:0') if cuda.is_available() else device('cpu')
        self.learning_rate = learnig_rate
        self.epoch = epoch
        self.interval_transform = {"days" : '1440', "minutes" : '1', "hours" : '60', "thrity" : "30"}
        setup_seed(20)
    
    def tuning_scheduler(self, epoch: int) -> float:
        self.learning_rate = self.learning_rate / (epoch + 1)
        return self.learning_rate
    
    def finetune_one(self, tickers: List[str], path: str):
        dataset = StockDataset(ticker=tickers, interval=self.interval_transform[self.interval])
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=1)
        
        input_sample, _ = dataset.__getitem__(0)
        model = ConvCausalLTSM(input_shape=input_sample.shape)
        
        model = train_model(
            model,
            dataloader,
            epochs=self.epoch,
            learning_rate=self.learning_rate,
            save_epoch=False,
            # lrfn=self.tuning_scheduler,
            lrfn=CosineWarmup(self.learning_rate, self.epoch).lrfn,
            # criterion=nn.L1Loss(), # l1 seems to make it harder
            checkpoint_file=load(self.base_weight)
        )

        save(model.state_dict(), path)
    
    def finetune_many(self):
        for group in TKGroup.__members__.values():
            group_name, tickers = group.value
            print(f'Fine-tuning for {group_name}: {tickers}')
            self.finetune_one(tickers, f'weight/{self.interval}/{group_name}.pth')

if __name__ == "__main__":
    # {"days" : '1440', "minutes" : '1', "hours" : '60', "thrity" : "30"}
    interval = "minutes"
    tuner = StockFineTuner(interval=interval, base_weight=f'weight/minutes/best_model.pth')
    tuner.finetune_many()
    # training is still realy weirdly reset while not in training phase
    # manage non crypto course via kraken
    # remove non crypto from inference 