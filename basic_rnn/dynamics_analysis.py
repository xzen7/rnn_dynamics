import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data_generator_2pointavg import MovingAverageDataset
from src.slow_point_finder import *
# Load the model
torch.load('./models/model_final.pth')
# Load the dataset and dataloader
dataset = MovingAverageDataset(sequence_length=10, total_samples=1000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# Extract activations
def extract_activations(model, dataloader, device):
    model.eval()
    activations = []  # Store activations here
    with torch.no_grad():
        for sequences, _ in dataloader:
            sequences = sequences.to(device)
            out, hidden = model(sequences)  # Assuming model returns hidden states and output
            activations.append(hidden.cpu().numpy())
    activations = np.concatenate(activations, axis=0)
    return activations
# TODO: calculate slow point

