# this file assumes the data is already downloaded and in the directory ../data

import os
from data_handler import DATA_DIR

vocab = set()
chvocab = set()

large = set()

file_list = os.listdir(DATA_DIR)
for i, file in enumerate(file_list):
  with open(os.path.join(DATA_DIR, file)) as f:
    text = f.read()
    f_vocab = set(text)
    currlen = len(vocab)
    vocab.update(f_vocab)
    print("{}/{} Added {} characters to vocab from {}".format(i+1, len(file_list), len(vocab) - currlen, file))

# vocab.remove('\n')
# vocab.remove(' ')

with open('./vocab.txt', 'w+') as f:
  f.write(''.join(list(vocab)))

print("{} total characters in vocab".format(len(vocab)))