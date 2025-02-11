import argparse
from torch import cuda, device, load, no_grad, tensor, nn
from torch.utils.data import DataLoader
from pandas import DataFrame

# Import necessary modules from your package
from hygdra_forecasting.utils.preprocessing import (
    ohlv_to_dataframe_inference, dataframe_to_dataset_inference
)
from hygdra_forecasting.model.build import ConvCausalLTSM, LtsmAttentionforecastPred
from hygdra_forecasting.model.build_graph import GraphforecastPred, GraphTransformerforecastPred

from hygdra_forecasting.dataloader.dataloader import StockDataset
from hygdra_forecasting.dataloader.GraphDataloader import StockGraphDataset
from hygdra_forecasting.model.train import train_model, setup_seed
from hygdra_forecasting.model.eval import validate
from hygdra_forecasting.utils.learning_rate_sheduler import CosineWarmup
from liquidnet.vision_liquidnet import VisionLiquidNet


def get_device():
    if cuda.is_available():
        print("Running on the GPU")
        return device("cuda:0")
    print("Running on the CPU")
    return device("cpu")

def load_model(model_name, input_shape, checkpoint_path=None):
    if model_name == "ConvCausalLTSM":
        model = ConvCausalLTSM(input_shape=input_shape)
    elif model_name == "LtsmAttentionforecastPred":
        model = LtsmAttentionforecastPred(input_shape=input_shape)
    elif model_name == "VisionLiquidNet":
        model = VisionLiquidNet(64, 10)
    elif model_name == "GraphTransformerforecastPred":
        model = GraphTransformerforecastPred(input_shape=input_shape)
    elif model_name == "GraphforecastPred":
        model = GraphforecastPred(input_shape=input_shape)
    else:
        raise ValueError("Unknown model type")
    
    if checkpoint_path:
        model.load_state_dict(load(checkpoint_path, weights_only=True))
        model.eval()
    return model

def inference(tickers):
    df, dict_unorm = ohlv_to_dataframe_inference(tickers)
    sequences_dict = dataframe_to_dataset_inference(df, tickers)
    input_shape = sequences_dict[tickers[0]][0].shape
    model = load_model("ConvCausalLTSM", input_shape, "weight/basemodel.pt")

    df_result = DataFrame()
    df_result.index = df[tickers[0] + "_close"].iloc[-100:].index
    for ticker in tickers:
        sequence = tensor(sequences_dict[ticker]).float()
        with no_grad():
            predictions = model(sequence)
        predictions = predictions.squeeze().numpy()
        df_result[ticker + "_pred"] = predictions.reshape(-1)[-100:] * dict_unorm[ticker][1] + dict_unorm[ticker][0]
        df_result[ticker + "_close"] = df[ticker + "_close"].iloc[-100:] * dict_unorm[ticker][1] + dict_unorm[ticker][0]
    
    print(df_result)

def evaluate(model_name, tickers):
    dataset = StockDataset(ticker=tickers)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=1)
    input_sample, _ = dataset.__getitem__(0)
    model = load_model(model_name, input_sample.shape, "weight/epoch-380_loss-0.2699198153614998.pt")
    criterion = nn.L1Loss()
    print(validate(model, dataloader, criterion))

def train(model_name, tickers, tickers_val, etf_tickers):
    if "graph" in model_name : 
        dataset = StockGraphDataset(ticker=tickers, indics=etf_tickers)
        dataset_val = StockGraphDataset(ticker=tickers_val, indics=etf_tickers)
    else :
        dataset = StockDataset(ticker=tickers)
        dataset_val = StockDataset(ticker=tickers_val)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)
    dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=True, num_workers=1)

    input_sample, _ = dataset.__getitem__(0)
    setup_seed(20)
    model = load_model(model_name, input_sample.shape)
    train_model(model, dataloader, val_dataloader=dataloader_val, epochs=100, learning_rate=0.01,
                lrfn=CosineWarmup(0.01, 100).lrfn)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["inference", "evaluate", "train"], required=True)
    parser.add_argument("--model", choices=["ConvCausalLTSM", "LtsmAttentionforecastPred", "GraphforecastPred", "GraphTransformerforecastPred"], required=True)
    parser.add_argument("--tickers", nargs='+', default=["DEFI", "PANW", "MRVL", "NKLA", "AFRM", "EBIT.TO", "^FCHI", "NKE", "^GSPC", "^IXIC", "BILL", "EXPE", 'LINK-USD', "TTWO", "NET", 'ICP-USD', 'FET-USD', 'FIL-USD', 'THETA-USD','AVAX-USD', 'HBAR-USD', 'UNI-USD', 'STX-USD', 'OM-USD', 'FTM-USD', "INJ-USD", "INTC", "SQ", "XOM", "COST", "BP", "BAC", "JPM", "GS", "CVX", "BA", "PFE", "PYPL", "SBUX", "DIS", "NFLX", 'GOOG', "NVDA", "JNJ", "META", "GOOGL", "AAPL", "MSFT", "BTC-EUR", "CRO-EUR", "ETH-USD", "CRO-USD", "BTC-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD"])
    parser.add_argument("--tickers_val", nargs='+', default=["AMZN", "AMD", "ETH-EUR", "ELF", "UBER"])
    parser.add_argument("--etf_tickers", nargs='+', default=["^GSPC", "^FCHI", "^IXIC","EBIT.TO", "BTC-USD"])
    args = parser.parse_args()

    if args.mode == "inference":
        inference(args.tickers)
    elif args.mode == "evaluate":
        evaluate(args.model, args.tickers)
    elif args.mode == "train":
        train(args.model, args.tickers, args.tickers_val, args.etf_tickers)

if __name__ == "__main__":
    main()

