from src.basic_rnn_model import SimpleRNN
from src.data_generator_2pointavg import MovingAverageDataset
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn

training_dataset = MovingAverageDataset(sequence_length=10, total_samples=1000)
# Split the dataset into training and validation and test sets
train_size = int(0.7 * len(training_dataset))
val_size = int(0.15 * len(training_dataset))
test_size = len(training_dataset) - train_size - val_size
training_dataset, validation_dataset, test_dataset = random_split(training_dataset, [train_size, val_size, test_size])
training_dataloader = DataLoader(training_dataset, batch_size=16, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# Hyper-parameters
input_size = 1
hidden_size = 32
output_size = 1
learning_rate = 0.001
num_epochs = 10

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = SimpleRNN(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
training_losses = []
validation_losses = []

total_step = len(training_dataloader)
for epoch in range(num_epochs):
    for i, (sequences, labels) in enumerate(training_dataloader):
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(sequences)[0]
        outputs = outputs.squeeze()
        training_loss = criterion(outputs, labels)
        training_losses.append(training_loss.item())

        # compute the validation loss
        model.eval()
        with torch.no_grad():
            for val_sequences, val_labels in validation_dataloader:
                val_sequences = val_sequences.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_sequences)[0]
                val_outputs = val_outputs.squeeze()
                val_loss = criterion(val_outputs, val_labels)
                validation_losses.append(val_loss.item())
        model.train()
        
        # Backward and optimize
        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()
        
        if (i+1) % 1 == 0: # every 1 steps
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, val_loss.item()))
    torch.save(model.state_dict(), './models/pointavgmodel_epoch'+str(epoch+1)+'_validationloss{:.7f}'.format(val_loss.item())+'.pth')
torch.save(model.state_dict(), './models/model_final.pth')

# compute the test loss
model.eval()
test_losses = []
with torch.no_grad():
    for test_sequences, test_labels in test_dataloader:
        test_sequences = test_sequences.to(device)
        test_labels = test_labels.to(device)
        test_outputs = model(test_sequences)[0]
        test_outputs = test_outputs.squeeze()
        test_loss = criterion(test_outputs, test_labels)
        test_losses.append(test_loss.item())
avg_test_loss = sum(test_losses) / len(test_losses)

# plot the loss
import matplotlib.pyplot as plt
import numpy as np
plt.figure()
plt.plot(training_losses, label='Training loss')    
plt.plot(validation_losses, label='Validation loss')
plt.legend()
plt.savefig('./training_figs/pointavgmodel_epoch'+str(epoch+1)+'_validationloss{:.7f}'.format(val_loss.item())+'_testloss{:.7f}'.format(avg_test_loss)+'.png')

