import torch
from icecream import ic
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import ImageFile

# Enable PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define a transform without Normalization
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

# Load the dataset and transform the image
dataset = datasets.ImageFolder(root="./data/CBIS_DDSM/", transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Calculate mean and standard deviation
mean = 0.0
std = 0.0
total_images_count = 0

for images, _ in loader:
    try:
        batch_size = images.size(0)
        # Flatten image pixels
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean(2).sum(0)  # Sum over batch and height*width
        std += images.std(2).sum(0)
        total_images_count += batch_size
    except Exception as e:
        print(f"Error processing image: {e}")

# Get the average pixel intensity over the dataset and average std
mean /= total_images_count
std /= total_images_count

print("Mean:", mean)
print("Standard Deviation:", std)