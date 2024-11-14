import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a temporary transform without Normalize
transform = transforms.Compose([
    transforms.Grayscale(),      # if your dataset is grayscale
    transforms.ToTensor()
])

# Load the dataset with this transform
dataset = datasets.ImageFolder(root="./data/CBIS_DDSM/", transform=transform) #./data/CBIS_DDSM/benign
loader = DataLoader(dataset, batch_size=1, shuffle=False) #, num_workers=4)

# Calculate mean and standard deviation
mean = 0.0
std = 0.0
total_images_count = 0

for images, _ in loader:
    batch_samples = images.size(0)  # batch size (the last batch can have fewer elements)
    images = images.view(batch_samples, images.size(1), -1)  # Flatten image pixels
    mean += images.mean(2).sum(0)  # Sum over batch and height*width
    std += images.std(2).sum(0)
    total_images_count += batch_samples

mean /= total_images_count
std /= total_images_count

print("Mean:", mean)
print("Standard Deviation:", std)