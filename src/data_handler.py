import os
import random

import torch
from torch.autograd import Variable

CHUNK_LEN = 200
DATA_DIR = "../data" # Where the training data is stored

with open('./vocab.txt', 'r') as f:
    vocab = f.read()

def char_tensor(s: str):
    ## tensor is a array
    tensor = torch.zeros(len(s)).long()
    for c in range(len(s)):
        try:
            tensor[c] = vocab.index(s[c])
        except ValueError:
            tensor[c] = 0 # if not in vocab, use 0 as index. 0 indexes a space in VOCAB
    return Variable(tensor)

# Update to random_chunk function to get a random chunk from a random file
def random_chunk():
    # generates a list of files in the training directory and chooses one randomly
    file_list = os.listdir(DATA_DIR)
    file_name = file_list[random.randint(0, len(file_list) -1)]
    file_raw = open(os.path.join(DATA_DIR, file_name), 'r')
    
    # reads the file into a string and then chooses a random chunk of that string to return
    file = file_raw.read()
    while len(file) < CHUNK_LEN:
        file_name = file_list[random.randint(0, len(file_list) -1)]
        file_raw = open(os.path.join(DATA_DIR, file_name), 'r')
        file = file_raw.read()

    start_index = random.randint(0, len(file) - (CHUNK_LEN + 1))
    end_index = start_index + CHUNK_LEN + 1
    ret = file[start_index:end_index]
    file_raw.close()
    return ret

# Finally we assemble a pair of input and target tensors for training, from a random chunk. The input will be all characters up to the end, and the target will be all characters from the first. So if our chunk is "abc" the input will correspond to "ab" while the target is "bc".
def random_training_set():    
    chunk = random_chunk()
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target

test_data = [
    ('Happ', 'y'),
    ('Happy Ne', 'w'),
    ('Happy New Yea', 'r'),
    ('That’s one small ste', 'p'),
    ('That’s one sm', 'a'),
    ('That’', 's'),
    ('Th', 'a'),
    ('one giant leap for mankin', 'd'),
    ('one giant leap fo', 'r'),
    ('one giant lea', 'p'),
    ('one giant l', 'e'),
    ('one gia', 't'),
    ('on', 'e'),
]