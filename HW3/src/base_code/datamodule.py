import re
import pickle
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader


class PAFDatamodule(LightningDataModule):
    def __init__(self, root_path, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.root = root_path
        self.classes = pickle.load(
            open(f"{root_path}/selected_families.pkl", "rb"))

    def encode_classes(self, y):
        cls2idx = dict(zip(self.classes, range(len(self.classes))))
        return [cls2idx[i] for i in y]

    def get_dataset(self, part, with_target=True):
        file_path = f"{self.root}/{part}_data.csv"
        df = pd.read_csv(file_path)
        x = df.loc[:, "sequence"].values
        # think about how to replace all rare/uncommon amino acids: X, U, B, O, Z
        if with_target:
            y = df.loc[:, "family_id"].values
            y = torch.tensor(self.encode_classes(y))
            x = list(zip(x, y))
        return x

    def train_dataloader(self):
        data = self.get_dataset("train")
        return DataLoader(data, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        data = self.get_dataset("val")
        return DataLoader(data, batch_size=self.batch_size, num_workers=4, shuffle=False)

    def test_dataloader(self):
        data = self.get_dataset("test")
        return DataLoader(data, batch_size=self.batch_size, num_workers=4, shuffle=False)

    def predict_dataloader(self):
        data = self.get_dataset("test", False)
        return DataLoader(data, batch_size=self.batch_size, num_workers=4, shuffle=False)
