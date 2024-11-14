import torch, torchvision
from torchvision import transforms
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import re
from icecream import ic
import glob

def load_mnist_dataloader(
        data_dir: str,
        image_size: int,
        batch_size: int = 16, 
        gpu : bool = True):

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

    test_dataset =  torchvision.datasets.MNIST(
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




#path to jpeg: "C:\Users\KenzaGarreau\PycharmProjects\AROB-Norse-CSNN\data\archive\jpeg"
#path to csv_file = "C:\Users\KenzaGarreau\PycharmProjects\AROB-Norse-CSNN\data\archive\csv\calc_case_description_train_set.csv"

# Define a function to extract the part of the path you want
def extract_id(path):
    # Use regular expression to find the correct segment
    match = re.search(r'\/([^\/]+)\/[^\/]+$', path)
    if match:
        return match.group(1)
    else:
        return path  # Return the original if pattern is not found

class CBISDDSM(Dataset):
    def __init__(self, data_dir, csv_file, transform=None, save_images=True):
        """
        Custom Dataset class for CBIS-DDSM images.

        Parameters:
        ----------------
        :data_dir: Directory where the CBIS-DDSM dataset folders are located.
        :csv_file: Path to the CSV file containing folder IDs and labels.
        :transform: Transformations to apply to the images.
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.save_images = save_images

        # Load the CSV file
        self.labels_df = pd.read_csv(csv_file)
        self.labels_df['pathology'] = self.labels_df['pathology'].str.lower()
        self.labels_df['image file path'] = self.labels_df['image file path'].apply(extract_id)
        ic(self.labels_df['image file path'][0])
        # Create a dictionary to map folder IDs to labels
        self.labels_dict = dict(zip(self.labels_df['image file path'], self.labels_df['pathology']))

        # Define the base directory for saving images
        self.save_dir = Path("./data/CBIS_DDSM")

        # Create directories for each category if saving is enabled
        if self.save_images:
            for category in ["malignant", "benign", "benign_without_callback"]:
                (self.save_dir / category).mkdir(parents=True, exist_ok=True)


        # Get list of folder paths (using folder names from 'id' column in CSV)
        self.folder_paths = [self.data_dir / folder_id for folder_id in self.labels_dict.keys()]

    def __len__(self):
        return len(self.folder_paths)

    def __getitem__(self, idx):

        folder_path = self.folder_paths[idx]
        folder_path = Path(
            folder_path)
        files = os.listdir(folder_path)
        #ic(files)
        #if os.path.exists(folder_path):
            #print("Path exists")
        # Get the first .jpeg or .jpg image in the folder
        image_files = glob.glob(f"{folder_path}/*.jpg")

        #print(f"Folder: {folder_path}, Images found: {(image_files)}")

        if len(image_files) == 0:
            raise FileNotFoundError(f"No JPEG image found in folder: {folder_path}")

        # Open the image
        image = Image.open(image_files[0]).convert("RGB")

        # Extract label from the labels dictionary
        folder_id = folder_path.name  # Folder name as the ID
        #label = 1 if self.labels_dict[folder_id] == "malignant" else 0
        pathology = self.labels_dict[folder_id]
        if pathology == "malignant":
            category = "malignant"
            label = 1
        elif pathology == "benign":
            category = "benign"
            label = 0
        elif pathology == "benign_without_callback":
            category = "benign_without_callback" #benign

        if self.save_images:
            dest_path = self.save_dir / category / f"{folder_id}.jpg"
            if not dest_path.exists():
                image.save(dest_path)

        if self.transform:
            image = self.transform(image)

        return image, label


def load_cbisdssm_dataloader(
        data_dir: str,
        csv_file: str,
        image_size: int,
        batch_size: int = 16,
        gpu: bool = True
):
    """
    Retrieves the CBIS-DDSM Dataset and
    returns torch dataloader for training and testing.

    Parameters :
    ----------------
    :data_dir: Path to the CBIS-DDSM dataset folders
    :csv_file: Path to the CSV file with folder IDs and labels
    :image_size: The input image size
    :batch_size: The batch size
    :gpu: Whether or not the GPU is used

    Returns :
    ----------------
    :train_dataloader:
    :test_dataloader:
    """
    n_workers = gpu * 4 * torch.cuda.device_count()

    # Define transforms suitable for CBIS-DDSM
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((image_size, image_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.1973], std=[0.2510]),
    ])

    # Create dataset instances for training and testing
    full_dataset = CBISDDSM(data_dir, csv_file, transform=transforms)

    # Optional split between train and test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=gpu,
        num_workers=n_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=gpu,
        num_workers=n_workers
    )

    return train_loader, test_loader