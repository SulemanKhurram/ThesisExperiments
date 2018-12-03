from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image

class OrigaDataset(Dataset):
    def __init__(self, file_path, transform = None):

        self.height = 224
        self.width = 224
        self.transform = transform
        self.lines = [line.rstrip('\n') for line in open(file_path)]
        self.paths = [line.split()[0] for line in self.lines]
        self.labels = []
        [self.labels.append(int(line.split()[1])) for line in self.lines]
        self.to_tensor = transforms.ToTensor()
        self.data_len = len(self.paths)

    def __getitem__(self, index):

        img = Image.open(self.paths[index])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[index]
        return img, label

    def __len__(self):
        return self.data_len  # of how many data(images?) you have

