import string
import os
import unidecode
import numpy as np

# This is the length of the pretrained embeddings being used
EMBEDDING_DIM = 300
EMBEDDINGS_PATH = "./embeddings.txt"
all_characters = string.printable

# Gets pretrained character embeddings from the path given at the top of the file
# Returns a np matrix in the order of string.printable with all the given embeddings
# Matrix has random embeddings if the given file didn't have an embedding for a charater
def get_embeddings():
    embeddings = {}
    file = open(EMBEDDINGS_PATH, 'r')
    for line in file:
        raw = line.strip().split()
        #First value in the line is the character name the rest are float values
        embedValues = np.asarray(raw[1:], dtype=float)
        char = raw[0]

        embeddings[char] = embedValues

    # Empty matrix going to be ordered by string.printable
    orderedWeights = np.zeros((len(all_characters), EMBEDDING_DIM))
    
    for i, char in enumerate(all_characters):
        try:
            # Uses the given embedding if present
            orderedWeights[i] = embeddings[char]
        except KeyError:
            # Generates a random embedding if not
            orderedWeights[i] = np.random.normal(scale=0.6, size=embedding_dim)

    return orderedWeights





