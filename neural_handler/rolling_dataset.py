from torch.utils.data import Dataset


class RollingDataset(Dataset):
    def __init__(self, items, seq_len):
        self.items = items
        self.seq_len = seq_len

    def __len__(self):
        return len(self.items) - self.seq_len

    def __getitem__(self, idx):
        # return (torch.tensor(self.features[idx], dtype=torch.float32),
        #         torch.tensor(self.labels[idx], dtype=torch.long))
        return self.items[idx: idx + self.seq_len], self.items[idx+1: idx + self.seq_len + 1]
