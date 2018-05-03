import random
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# loader for training (maybe apply data augmentation
train_transformer = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

# loader for validation
val_transformer = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor()])


class ClimateDataset(Dataset):

    def __init__(self, data_dir, transform):

        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.jpg')]

        self.labels = [int(filename.split('/')[-1][0]) for filename in self.filenames]
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):

        image = Image.open(self.filenames[idx])  # PIL image
        image = self.transform(image)
        return image, self.labels[idx]


def fetch_dataloader(types, data_dir, batch_size):

    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:

            path = os.path.join(data_dir, "{}".format(split))

            if split == 'train':
                dl = DataLoader(ClimateDataset(path, train_transformer), batch_size=batch_size, shuffle=True)
            else:
                dl = DataLoader(ClimateDataset(path, val_transformer), batch_size=batch_size, shuffle=False)

            dataloaders[split] = dl

    return dataloaders
