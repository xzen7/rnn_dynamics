from src.basic_rnn_model import SimpleRNN
from src.data_generator_2pointavg import MovingAverageDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

dataset = MovingAverageDataset(sequence_length=10, total_samples=1000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# # test the dataset and dataloader
# for i, (sequences, labels) in enumerate(dataloader):
#     print(sequences, labels)
#     break

# Hyper-parameters
input_size = 1
hidden_size = 32
output_size = 1
learning_rate = 0.001
num_epochs = 2

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = SimpleRNN(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(dataloader)
for epoch in range(num_epochs):
    for i, (sequences, labels) in enumerate(dataloader):
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(sequences)
        
        outputs = outputs.squeeze()
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 1 == 0: # every 1 steps
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    torch.save(model.state_dict(), './models/model_epoch'+str(epoch+1)+'_loss{:.4f}'.format(loss.item())+'.pth')
