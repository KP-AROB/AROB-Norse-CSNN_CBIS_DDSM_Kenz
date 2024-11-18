import pandas as pd
import numpy as np
from neuroCombat import neuroCombat

import torch, torchvision
from torchvision import transforms
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from src.utils.dataloaders_Kenza import load_cbisdssm_dataloader
import pandas as pd
import os
import re
from icecream import ic
import glob


def load_mnist_dataloader(
        data_dir: str,
        image_size: int,
        batch_size: int = 16,
        gpu: bool = True):
    """
    Retrieves the MNIST Dataset and
    returns torch dataloader for training and testing

    Parameters :
    ----------------

    :image_size: The input image size
    :batch_size: The batch size
    :gpu: Whether or not the gpu is used

    Returns :
    ----------------
    :train_dataloader:
    :test_dataloader:

    """
    n_workers = gpu * 4 * torch.cuda.device_count()

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms,
    )

    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=gpu,
        num_workers=n_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=gpu,
        num_workers=n_workers
    )

    return train_loader, test_loader


# Define a function to extract the part of the path you want
def extract_id(path):
    # Use regular expression (re) to find the string segment we want
    match = re.search(r'\/([^\/]+)\/[^\/]+$', path)
    if match:
        return match.group(1)
    else:
        return path  # Return the original path if the segment is not found


class CBI_forCombat(Dataset):
    def __init__(self, data_dir, csv_file, transform=None, save_images=False):
        """
        Custom Dataset class for CBIS-DDSM images.

        Parameters:
        ----------------
        :data_dir: Directory where the CBIS-DDSM dataset folders are located.
        :csv_file: Path to the CSV file containing folder names and labels.
        :transform: Transformations to apply to the images.
        :save_images: Whether or not to save the images in the CBIS_DDSM folder.
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.save_images = save_images

        # Load the CSV file
        self.labels_df = pd.read_csv(csv_file)

        # Extract the desired columns
        columns_to_extract = ['image file path', 'pathology', 'left or right breast']
        extracted_df = self.labels_df[columns_to_extract]
        print(extracted_df)

        self.labels_df['pathology'] = self.labels_df['pathology'].str.lower()
        self.labels_df['image file path'] = self.labels_df['image file path'].apply(extract_id)
        #self.labels_df['left or right breast']
        ic(self.labels_df['image file path'][0])
        # Create a dictionary to map folder image file path to labels
        #self.labels_dict = dict(zip(self.labels_df['image file path'], self.labels_df['pathology']))

        # Define the directory for saving images
        self.save_dir = Path("./data/CBIS_DDSM")

        # Add directories for each class if saving is enabled
        if self.save_images:
            for category in ["malignant", "benign", "benign_without_callback"]:
                (self.save_dir / category).mkdir(parents=True, exist_ok=True)

        self.labels_dict = dict(zip(self.labels_df['image file path'], self.labels_df['pathology']))

        # Create a list of folder paths (using folder names from 'image file path' column in CSV)
        self.folder_paths = [self.data_dir / folder_id for folder_id in self.labels_dict.keys()]

    def __len__(self):
        return len(self.folder_paths)

    # the method getitem is called by the dataloader
    def __getitem__(self, idx):

        folder_path = self.folder_paths[idx]
        folder_path = Path(
            folder_path)

        # Get the first .jpeg or .jpg image in the folder
        image_files = glob.glob(f"{folder_path}/*.jpg")

        # print(f"Folder: {folder_path}, Images found: {(image_files)}")

        if len(image_files) == 0:
            raise FileNotFoundError(f"No JPEG image found in folder: {folder_path}")

        # open the image
        image = Image.open(image_files[0])  # .convert("RGB")

        # Assign label to the classes dictionary
        folder_id = folder_path.name  # Folder name as the ID
        pathology = self.labels_dict[folder_id]
        if pathology == "malignant":
            category = "malignant"
            label = 1
        elif pathology == "benign":
            category = "benign"
            label = 0
        elif pathology == "benign_without_callback":
            category = "benign_without_callback"  # benign
            label = 2

        if self.save_images:
            dest_path = self.save_dir / category / f"{folder_id}.jpg"
            if not dest_path.exists():
                image.save(dest_path)

        if self.transform:
            image = self.transform(image)

        return image, label


def correction(labels_df, image_data):
    # Reshape image
    image_data_array = np.array([sample[0].flatten() for sample in image_data])
    image_data_T = image_data_array.T
    # define the confounder
    confounder = labels_df['left or right breast'].values

    # list of covariates
    covars = pd.DataFrame({
        'confounder': confounder,
        'label': labels_df['pathology'].values,
        'ID': labels_df['image file path'].values
    })

    # Run ComBat
    combat_corrected = neuroCombat(
        dat=image_data_T,
        covars=covars,
        batch_col='confounder')['data']

    print("Corrected data shape:", combat_corrected)
    return combat_corrected


def load_cbisdssm_dataloader(
        data_dir: str,
        csv_file: str,
        image_size: int,
        batch_size: int = 16,
        gpu: bool = True,
        n_classes=3,
):
    """
    Retrieves the CBIS-DDSM Dataset and
    returns torch dataloader for training and testing.

    Parameters :
    ----------------
    :data_dir: Path to the CBIS-DDSM dataset folders
    :csv_file: Path to the CSV file with folder "image paths" and labels
    :image_size: The input image size
    :batch_size: The batch size
    :gpu: Whether or not the GPU is used

    Returns :
    ----------------
    :train_dataloader:
    :test_dataloader:
    """
    n_workers = gpu * 4 * torch.cuda.device_count()

    # Define list of transforms for CBIS-DDSM images
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((image_size, image_size)),
        torchvision.transforms.ToTensor(),
        # Normalisation values for CBIS_DDSM
        torchvision.transforms.Normalize(mean=[0.2016], std=[0.2540]),  # (mean=[0.1973], std=[0.2510]),
        # Normalisation values for MNIST ((0.1307,), (0.3081,))
        # torchvision.transforms.RandomHorizontalFlip(p=0.1),
        torchvision.transforms.RandomVerticalFlip(p=0.1),
        # torchvision.transforms.ElasticTransform()
    ])

    # Create dataset instances
    full_dataset = CBI_forCombat(data_dir, csv_file, transform=transforms)
    # Split between train and test
    train_size = int(0.8 * len(full_dataset))  # Lenth of folder_paths
    ic(train_size)
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    combated_train_dataset = correction(full_dataset.labels_df, train_dataset)
    combated_test_dataset = correction(full_dataset.labels_df, test_dataset)

    # Create the Dataloader, the Dataloader calls automatically the method __getitem__
    train_loader = DataLoader(
        combated_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=gpu,
        num_workers=n_workers
    )

    test_loader = DataLoader(
        combated_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=gpu,
        num_workers=n_workers
    )

    return train_loader, test_loader, n_classes
