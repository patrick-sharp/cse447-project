import string

import torch
import torch.nn as nn
from torch.autograd import Variable

HIDDEN_SIZE = 139
NUM_LAYERS = 6
VOCAB = ' ' + string.ascii_letters
NUM_CHARACTERS = len(VOCAB)

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
    
    # Turn string into list of longs
    @classmethod
    def char_tensor(cls, s: str):
        ## tensor is a array
        tensor = torch.zeros(len(s)).long()
        for c in range(len(s)):
            try:
                tensor[c] = VOCAB.index(s[c])
            except ValueError:
                tensor[c] = 0 # if not in vocab, use 0 as index. 0 indexes a space in VOCAB
        return Variable(tensor)
    
    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
    
    def predict(self, history='A', num_top_choices=3, temperature=0.8):
        hidden = self.init_hidden()
        history_input = Model.char_tensor(history)

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
            predicted_chars.append(VOCAB[i])
        return predicted_chars