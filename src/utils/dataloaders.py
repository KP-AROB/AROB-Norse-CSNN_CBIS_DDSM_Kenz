import torch, torchvision
from torch.utils.data import random_split
from torchvision import transforms

def load_image_folder_dataloader(
        data_dir: str,
        image_size: int,
        batch_size: int = 16,
        gpu : bool = True):
    
    n_workers = gpu * 4 * torch.cuda.device_count()

    dataset = torchvision.datasets.ImageFolder(
        data_dir, 
        transform=transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(), 
                transforms.Resize((image_size, image_size)),
                transforms.Normalize((0.1973),(0.2510)),#this is for CBIS_DDSM #((0.1307,), (0.3081,)),#.2510 and  #maybe have to change for the
                #transforms.RandomHorizontalFlip(p=0.1),
                transforms.RandomVerticalFlip(p=0.1),
                #transforms.ElasticTransform()
                #  transforms.RandomResizedCrop(size=(224, 224), antialias=True),

            ]
        )
    )

    n_classes = len(dataset.classes)
    print("\n# Available labels in dataset :", dataset.class_to_idx)

    img_nums = int(len(dataset))
    valid_num = int(img_nums * 0.2)
    train_num = img_nums - valid_num
    train_data, val_data = random_split(dataset, [train_num, valid_num])

    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=gpu,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=gpu,
    )
    
    return train_dataloader, val_dataloader, n_classes

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
    
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(image_size), 
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(image_size), 
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    print("\n# Available labels in dataset :", train_dataset.class_to_idx)

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

    n_classes = len(train_dataset.classes)

    return train_loader, test_loader, n_classes