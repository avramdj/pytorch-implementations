import pandas as pd
from torch.utils.data import Dataset

MAX_SEQ_LEN = 512


class ImdbDataset(Dataset):
    def __init__(self, file_name):
        self.df = pd.read_csv(file_name)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.review[idx][:MAX_SEQ_LEN], float(
            self.df.sentiment[idx] == "positive"
        )
