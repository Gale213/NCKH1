import torch
from torch.utils.data import Dataset, DataLoader

class CustomGraphDataset(Dataset):
    def __init__(root, file_path):
        # Load your graph dataset from file
        root.data = 'C:\\NCKH\\Graph\\Graph\\Trojan'  # Load your dataset from data_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a single sample (x, adj, label) from the dataset
        sample = self.data[idx]
        return sample['features'], sample['adjacency_matrix'], sample['label']

# Define the path to your dataset file
data_path = 'C:\\NCKH\\Graph\\Graph\\Trojan'

# Initialize your custom dataset
dataset = CustomGraphDataset('C:\\NCKH\\Graph\\Graph\\Trojan')

# Create DataLoader
batch_size = 64
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
