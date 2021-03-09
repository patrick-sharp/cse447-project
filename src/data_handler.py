import os
import random

from tqdm import tqdm
import torch
from torch.autograd import Variable

CHUNK_LEN = 400
DATA_DIR = "../data" # Where the training data is stored
CURR_DIR = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(CURR_DIR, 'vocab.txt'), 'r') as f:
    vocab = f.read()

def char_tensor(s: str):
    tensor = torch.zeros(len(s)).long()
    for c in range(len(s)):
        try:
            tensor[c] = vocab.index(s[c])
        except ValueError:
            tensor[c] = 0 # if not in vocab, use 0 as index. 0 indexes the null symbol in vocab
    return Variable(tensor)

# Update to random_chunk function to get a random chunk from a random file
# def random_chunk():
#     # generates a list of files in the training directory and chooses one randomly
#     file_list = os.listdir(DATA_DIR)
#     file_name = file_list[random.randint(0, len(file_list) -1)]
#     file_raw = open(os.path.join(DATA_DIR, file_name), 'r')
    
#     # reads the file into a string and then chooses a random chunk of that string to return
#     file = file_raw.read()
#     while len(file) < CHUNK_LEN:
#         file_name = file_list[random.randint(0, len(file_list) -1)]
#         file_raw = open(os.path.join(DATA_DIR, file_name), 'r')
#         file = file_raw.read()

#     start_index = random.randint(0, len(file) - (CHUNK_LEN + 1))
#     end_index = start_index + CHUNK_LEN + 1
#     ret = file[start_index:end_index]
#     file_raw.close()
#     return ret

# Finally we assemble a pair of input and target tensors for training, from a random chunk. The input will be all characters up to the end, and the target will be all characters from the first. So if our chunk is "abc" the input will correspond to "ab" while the target is "bc".
# def random_training_set():    
#     chunk = random_chunk()
#     inp = char_tensor(chunk[:-1])
#     target = char_tensor(chunk[1:])
#     return inp, target

class TrainData():
    def __init__(self, batch_size=2000):    
        self.batch_index = 0
        self.batch_size = batch_size
        inp = []
        file_list = os.listdir(DATA_DIR)
        non_cuda_tensors = 0
        for filename in tqdm(file_list):
            with open(os.path.join(DATA_DIR, filename), 'r') as f:
                for line in f:
                    line = line.strip('\n')
                    if len(line) <= 1:
                        continue
                    tensor = char_tensor(line)
                    tensor.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                    if not tensor.is_cuda:
                        non_cuda_tensors += 1
                    inp.append(tensor)
        random.shuffle(inp)
        print(f'Non-cuda tensors in dataset: {non_cuda_tensors}')
        self.inp = inp
    def random_training_set(self):
        if self.batch_index + self.batch_size >= len(self.inp):
            return self.inp[self.batch_index:]
            self.batch_index = 0
        else:
            return self.inp[self.batch_index:self.batch_index + self.batch_size]
            self.batch_index += self.batch_size

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
    ('one gia', 'n'),
    ('on', 'e'),
    ('Ruin h', 'a'),
    ('Ruin has co', 'm'),
    ('Ruin has come to o', 'u'),
    ('Ruin has come to our fam', 'i'),
    ('Ruin has come to our famil', 'y'),
    ('You rem', 'e'),
    ('You remember our vene', 'r'),
    ('You remember our venerabl', 'e'),
    ('You remember our venerabl hou', 's'),
    ('You remember our venerabl house, op', 'u'),
    ('You remember our venerabl house, opulent and im', 'p'),
    ('You remember our venerabl house, opulent and imperia', 'l'),
    ('G', 'a'),
    ('Gaz', 'i'),
    ('Gazin', 'g'),
    ('Gazing pro', 'u'),
    ('Gazing proudl', 'y'),
    ('Gazing proudly from its sto', 'i'),
    ('Gazing proudly from its stoic per', 'c'),
    ('Gazing proudly from its stoic perch ab', 'o'),
    ('Gazing proudly from its stoic perch abo', 'v'),
    ('Gazing proudly from its stoic perch above the moo', 'r'),
    ('Mons', 't'),
    ('Monst', 'r'),
    ('Monstrou', 's'),
    ('Monstrous si', 'z'),
    ('Monstrous size has no in', 't'),
    ('Monstrous size has no intrins', 'i'),
    ('Monstrous size has no intrinsic mer', 'i'),
    ('Monstrous size has no intrinsic meri', 't'),
    ('Monstrous size has no intrinsic merit, unl', 'e'),
    ('Monstrous size has no intrinsic merit, unless inor', 'd'),
    ('Monstrous size has no intrinsic merit, unless inordinat', 'e'),
    ('Monstrous size has no intrinsic merit, unless inordinate e', 'x'),
    ('Monstrous size has no intrinsic merit, unless inordinate exsan', 'g'),
    ('Monstrous size has no intrinsic merit, unless inordinate exsanguinatio', 'n'),
    ('Monstrous size has no intrinsic merit, unless inordinate exsanguination be co', 'n'),
    ('Monstrous size has no intrinsic merit, unless inordinate exsanguination be con', 's'),
    ('Monstrous size has no intrinsic merit, unless inordinate exsanguination be consid', 'e'),
    ('Monstrous size has no intrinsic merit, unless inordinate exsanguination be conside', 'r'),
    ('Monstrous size has no intrinsic merit, unless inordinate exsanguination be consider', 'e'),
    ('Monstrous size has no intrinsic merit, unless inordinate exsanguination be considere', 'd'),
    ('Monstrous size has no intrinsic merit, unless inordinate exsanguination be considered a vir', 't'),
    ('Monstrous size has no intrinsic merit, unless inordinate exsanguination be considered a virt', 'u'),
    ('Monstrous size has no intrinsic merit, unless inordinate exsanguination be considered a virtu', 'e'),
]

