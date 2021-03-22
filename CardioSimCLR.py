from torchvision.transforms import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from data_aug.gaussian_blur import GaussianBlur
from data_aug.view_generator import ContrastiveLearningViewGenerator
import numpy as np 
import torch
from PIL import Image

class CardioSimCLRDataset(Dataset):
    
    """Dataset of images. Returns unlabelled dataset for contrastive learning"""
    def __init__(self, npz_file, transform=None, to_tensor=False, to_normalize = False):
        """
        Args:
            npz_file (string): Path to the npz file with patches and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.patches, self.labels = load_data(npz_file)
        self.transform = transform
        self.tensor = to_tensor
        self.norm = to_normalize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.patches[idx]
        img = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(img)

        sample = (image, -1) 
    
        return sample

def get_simclr_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.Resize(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

def load_data(file_path): 
    
    """Load training data from file in ``.npz`` format."""
    f = np.load(file_path, allow_pickle=True)
    X, Y = f['X'], f['Y']
    Y=np.squeeze(Y)
    return (X,Y)

def get_cardio_smclr(train_file):
    
    train_data = CardioSimCLRDataset(train_file, transform = ContrastiveLearningViewGenerator(get_simclr_transform(96), 2))
    
    return train_data
