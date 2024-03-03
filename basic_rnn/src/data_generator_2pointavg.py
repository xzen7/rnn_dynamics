from torch.utils.data import Dataset, DataLoader
import numpy as np

class MovingAverageDataset(Dataset):
    def __init__(self, sequence_length=100, total_samples=1000):
        self.sequence_length = sequence_length
        self.total_samples = total_samples
        self.data = self._generate_data()
    
    def _generate_data(self):
        # Generate random sequences
        data = np.random.rand(self.total_samples, self.sequence_length)
        # Calculate the moving average
        labels = np.mean(data[:, -2:], axis=1)  # 2-point moving average of the last two numbers
        return data, labels
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        sample = self.data[0][idx].reshape(-1, 1)  # Reshape for RNN input
        label = self.data[1][idx]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

