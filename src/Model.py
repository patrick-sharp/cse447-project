import string

import torch
import torch.nn as nn
from torch.autograd import Variable
from data_handler import CHUNK_LEN, vocab, char_tensor, test_data

HIDDEN_SIZE = 139
NUM_LAYERS = 6
# VOCAB = ' ' + string.ascii_letters
NUM_CHARACTERS = len(vocab)

class Model(nn.Module):
    def __init__(self, input_size=NUM_CHARACTERS, hidden_size=HIDDEN_SIZE, output_size=NUM_CHARACTERS, n_layers=NUM_LAYERS):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
    
    def train(self, inp, target, criterion, optim,):
        hidden = self.init_hidden()
        self.zero_grad()
        loss = 0
        for c in range(CHUNK_LEN):
            output, hidden = self(inp[c], hidden)
            '''unsqueeze() is used to add dimension to the tensor'''
            loss += criterion(output, target[c].unsqueeze(dim=0))
        # Back propagation
        loss.backward()
        optim.step()
        return loss.item() / CHUNK_LEN
    
    def predict(self, history='A', num_top_choices=3, temperature=0.8):
        hidden = self.init_hidden()
        history_input = char_tensor(history)

        # Use priming string to "build up" hidden state
        for c in range(len(history) - 1):
            _, hidden = self(history_input[c], hidden)
        inp = history_input[-1]

        output, hidden = self(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, num_top_choices)
        
        # Add predicted character to string and use as next input
        predicted_chars = []
        for i in top_i:
            predicted_chars.append(vocab[i])
        return predicted_chars
    
    # test data is a tuple of strings. first string is history, next string is correct char
    def evaluate(self, test_data=test_data):
        total = len(test_data)
        correct = 0
        for history, next_char in test_data:
            preds = self.predict(history)
            if next_char in preds:
                correct += 1
        return correct / total

