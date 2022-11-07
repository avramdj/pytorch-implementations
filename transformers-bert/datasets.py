import pandas as pd
from torch.utils.data import Dataset


class ImdbDataset(Dataset):
    def __init__(self, file_name):
        self.df = pd.read_csv(file_name)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.review[idx][:512], float(self.df.sentiment[idx] == "positive")
