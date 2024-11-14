import torch, random, os
import numpy as np
from argparse import ArgumentParser
from src.utils.dataloaders import load_mnist_dataloader, load_image_folder_dataloader
from src.models.classification import SimpleCLSModel
from src.utils.decoders import softmax_decoder
from src.networks.classification import ConvNet
from norse.torch import ConstantCurrentLIFEncoder
from src.experiment.classification import ClassificationExperiment
from torch.utils.tensorboard import SummaryWriter
from uuid import uuid4
from src.utils.parameters import write_params_to_file, load_parameters

if __name__ == "__main__":

    parser = ArgumentParser()
    #parser.add_argument("--params", type=str, default='./params/mnist.yaml')
    parser.add_argument("--params", type=str, default='./params/inbreast.yaml')
    args = parser.parse_args()
    params = load_parameters(args.params)

    ## ========== INIT ========== ##

    gpu = torch.cuda.is_available()

    if gpu:
        torch.cuda.manual_seed_all(params["seed"])
    else:
        torch.manual_seed(params["seed"])
    random.seed(params["seed"])
    np.random.seed(params["seed"])

    experiment_id = str(uuid4())[:8]
    experiment_name = f'experiment_{experiment_id}' if not params['name'] else f"{params['name']} - {experiment_id}"
    print('\n# Initialization of the experiment protocol - {} \n'.format(experiment_name))
    log_dir = os.path.join(params["log_dir"], experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    write_params_to_file(vars(args), log_dir)
    log_interval = max(100 // params["batch_size"], 1)

    # ========== DATALOADER ========== ##

    if params['task'] == 'MNIST':
        train_dl, test_dl, n_classes = load_mnist_dataloader(
            params["data_dir"], 
            params["input_size"], 
            params["batch_size"], 
            gpu)
    else:
        train_dl, test_dl, n_classes = load_image_folder_dataloader(
            params["data_dir"], 
            params["input_size"], 
            params["batch_size"], 
            gpu)

    print('# Dataloaders successfully loaded.\n')

    ## ========== MODEL ========== ##

    DEVICE = torch.device("cuda") if gpu else torch.device("cpu")

    model = SimpleCLSModel(
        encoder=ConstantCurrentLIFEncoder(seq_length=params["max_latency"]), 
        snn=ConvNet(alpha=80, n_classes = n_classes, feature_size=params["input_size"]), 
        decoder=softmax_decoder
    ).to(DEVICE)

    ## ========== TRAINING ========== ##

    writer = SummaryWriter(log_dir=log_dir, flush_secs=60)

    print(f'Running on device : {DEVICE}')
    experiment = ClassificationExperiment(
        model=model,
        writer=writer,
        lr=params["lr"], 
        log_interval=log_interval,
        device=DEVICE)
    
    experiment.fit(train_dl, test_dl, params["epochs"])
