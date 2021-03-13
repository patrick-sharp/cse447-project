# this file assumes the data is already downloaded and in the directory ../data

import os

DATA_DIR = "../data" # Where the training data is stored

vocab = set()

file_list = os.listdir(DATA_DIR)
for i, file in enumerate(file_list):
  with open(os.path.join(DATA_DIR, file)) as f:
    text = f.read()
    f_vocab = set(text)
    currlen = len(vocab)
    vocab.update(f_vocab)
    print("{}/{} Added {} characters to vocab from {}".format(i+1, len(file_list), len(vocab) - currlen, file))

voc_list = list(vocab)
voc_list.insert(0, '\u2400')
voc_string = ''.join(voc_list)

with open('./vocab.txt', 'w+') as f:
  f.write(voc_string)

print("{} total characters in vocab".format(len(vocab)))