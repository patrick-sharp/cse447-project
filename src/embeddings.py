import string
import os
import numpy as np
import torch

# This is the length of the pretrained embeddings being used
EMBEDDING_DIM = 300
EMBEDDINGS_PATH = "./embeddings.txt"

# Gets pretrained character embeddings from the path given at the top of the file
# Returns a np matrix in the order of string.printable with all the given embeddings
# Matrix has random embeddings if the given file didn't have an embedding for a charater
def get_tensor_embeddings(input_size, hidden_size, vocab):
    embeddings = {}
    file = open(EMBEDDINGS_PATH, 'r')
    for line in file:
        raw = line.strip().split()
        #First value in the line is the character name the rest are float values
        embedValues = np.asarray(raw[1:], dtype=float)
        char = raw[0]

        embeddings[char] = embedValues
    
    tensor_embeddings = torch.normal(0,1,(input_size, hidden_size))
    for i, char in enumerate(vocab):
        if char in embeddings:
            tensor_embeddings[i] = torch.Tensor(embeddings[char])

    return tensor_embeddings
