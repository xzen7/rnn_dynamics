import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate RNN
        out, hn = self.rnn(x, h0)
        
        # Pass the output of the last time step to the classifier
        out = self.linear(out[:, -1, :])
        return out, hn.squeeze(0) # squeezed to remove the additional dimension added by the batch
