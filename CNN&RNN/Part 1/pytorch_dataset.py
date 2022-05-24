from torch.utils.data import Dataset
from sklearn.datasets import make_moons
from sklearn.preprocessing import OneHotEncoder
import torch


class PointDataset(Dataset):
    def __init__(self, n_samples=2000, point_set_type="moons"):
        self.x = None
        self.label = None
        if point_set_type == "moons":
            self.x, self.label = make_moons(n_samples)
        else:
            raise NotImplementedError

        self.label = OneHotEncoder(sparse=False).fit_transform(self.label.reshape(len(self.label), 1))
        self.x, self.label = torch.FloatTensor(self.x), torch.FloatTensor(self.label)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.label[idx]
