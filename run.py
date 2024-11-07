import torch
from argparse import ArgumentParser
from src.utils.dataloaders import load_mnist_dataloader
from src.models.classification import SimpleCLSModel
from src.utils.decoders import softmax_decoder
from src.networks.conv import ConvNet
from norse.torch import ConstantCurrentLIFEncoder
from src.experiment.classification import ClassificationExperiment

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='./data')
    parser.add_argument("--input_size", type=int, default=28)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_latency", type=int, default=50)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--epochs", type=int, default=1)

    args = parser.parse_args()

    gpu = torch.cuda.is_available()

    train_dl, test_dl = load_mnist_dataloader(
        args.data_dir, 
        args.input_size, 
        args.batch_size, 
        gpu)

    DEVICE = torch.device("cuda") if gpu else torch.device("cpu")

    model = SimpleCLSModel(
        encoder=ConstantCurrentLIFEncoder(seq_length=args.max_latency), 
        snn=ConvNet(alpha=80), 
        decoder=softmax_decoder
    ).to(DEVICE)

    print(f'Running on device : {DEVICE}')
    experiment = ClassificationExperiment(model, args.lr, DEVICE)
    experiment.fit(train_dl, test_dl, args.epochs)
