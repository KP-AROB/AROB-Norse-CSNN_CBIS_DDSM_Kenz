import torch, random, os
from torchvision.transforms import v2
from torchvision.io import decode_image
import numpy as np
from argparse import ArgumentParser
from src.utils.dataloaders_Kenza import load_mnist_dataloader, load_cbisdssm_dataloader
from src.utils.remove_confounder import load_cbisdssm_dataloader
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

    path_jpeg = "./data/archive/jpeg"
    #path_csv = "./data/archive/csv/calc_case_description_test_set.csv"
    path_csv = "./data/archive/csv/calc_case_description_train_set.csv"


    train_dl, test_dl, n_classes = load_cbisdssm_dataloader(
        data_dir=path_jpeg,
        csv_file=path_csv,
        image_size= params["input_size"],
        batch_size=params["batch_size"],
        gpu=torch.cuda.is_available(),
        #save_images = False,
    )


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
