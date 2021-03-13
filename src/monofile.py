import os
import random
import time
import math
import wikipedia
from google_trans_new import google_translator
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURR_DIR, '../data')
EMBEDDING_DIM = 300
EMBEDDINGS_PATH = os.path.join(CURR_DIR, "embeddings.txt")
HIDDEN_SIZE = 2048
NUM_LAYERS = 2
CHUNK_LEN = 400
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
TEST_DATA = test_data = [
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
    ('You remember our venerable hou', 's'),
    ('You remember our venerable house, op', 'u'),
    ('You remember our venerable house, opulent and im', 'p'),
    ('You remember our venerable house, opulent and imperia', 'l'),
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
random.seed(447)

def get_vocab():
  vocab = set()
  file_list = os.listdir(DATA_DIR)
  for i, file in enumerate(file_list):
    with open(os.path.join(DATA_DIR, file)) as f:
      text = f.read()
      f_vocab = set(text)
      currlen = len(vocab)
      vocab.update(f_vocab)
      # print("{}/{} Added {} characters to vocab from {}".format(i+1, len(file_list), len(vocab) - currlen, file))
  voc_list = list(vocab)
  voc_list.insert(0, '\u2400')
  voc_string = ''.join(voc_list)

  print("{} total characters in vocab".format(len(vocab)))
  return voc_string

with open(os.path.join(CURR_DIR, 'vocab.txt'), 'r') as f:
  vocab = f.read()
NUM_CHARACTERS = len(vocab)

def time_since(since):
  s = time.time() - since
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)
  
def char_tensor(s: str):
  if torch.cuda.is_available():
    tensor = torch.zeros(len(s)).to(torch.int64).cuda('cuda')
  else:
    tensor = torch.zeros(len(s)).to(torch.int64)
  for c in range(len(s)):
    try:
      tensor[c] = vocab.index(s[c])
    except ValueError:
      tensor[c] = 0 # if not in vocab, use 0 as index. 0 indexes the null symbol in vocab
  return Variable(tensor)

class TrainData():
  def __init__(self, batch_size=BATCH_SIZE):    
    self.batch_index = 0
    self.batch_size = batch_size
    data = []
    file_list = os.listdir(DATA_DIR)
    non_cuda_tensors = 0
    print(len(file_list), 'articles in the dataset')
    hundreth = len(file_list) // 100
    tenth = len(file_list) // 10
    for i, filename in enumerate(file_list):
      if (i+1) % tenth == 0:
        print('|', end='')
      elif (i+1) % hundreth == 0:
        print('-', end='')
      with open(os.path.join(DATA_DIR, filename), 'r') as f:
        for line in f:
          line = line.strip('\n')
          if len(line) <= 1:
            continue
          elif len(line) > CHUNK_LEN:
            idx = 0
            while idx < len(line):
              if idx + CHUNK_LEN < len(line):
                chunk = line[idx:idx+CHUNK_LEN]
              else:
                chunk = line[idx:]
              tensor = char_tensor(chunk)
              if not tensor.is_cuda:
                non_cuda_tensors += 1
              data.append(tensor)
              idx += CHUNK_LEN
          else:
            tensor = char_tensor(line)
            if not tensor.is_cuda:
              non_cuda_tensors += 1
            data.append(tensor)
    random.shuffle(data)
    print(f'\nNon-cuda tensors in dataset: {non_cuda_tensors}')
    print(f'Total tensors in dataset: {len(data)}')
    self.data = data
  def random_training_set(self):
    if self.batch_index + self.batch_size >= len(self.data):
      self.batch_index = 0
      return self.data[self.batch_index:]
    else:
      self.batch_index += self.batch_size
      return self.data[self.batch_index:self.batch_index + self.batch_size]

def get_tensor_embeddings(input_size, embedding_dim, vocab):
  embeddings = {}
  try:
    file = open(EMBEDDINGS_PATH, 'r')
    for line in file:
      raw = line.strip().split()
      #First value in the line is the character name the rest are float values
      embedValues = np.asarray(raw[1:], dtype=float)
      char = raw[0]
      embeddings[char] = embedValues
  except:
    pass
  
  if torch.cuda.is_available():
    tensor_embeddings = torch.normal(0,1,(input_size, embedding_dim)).cuda('cuda')
  else:
    tensor_embeddings = torch.normal(0,1,(input_size, embedding_dim))
  for i, char in enumerate(vocab):
    if char in embeddings:
      tensor_embeddings[i] = torch.Tensor(embeddings[char])
  return tensor_embeddings

class Model(nn.Module):
  def __init__(self, vocab=vocab, input_size=NUM_CHARACTERS, hidden_size=HIDDEN_SIZE, output_size=NUM_CHARACTERS, n_layers=NUM_LAYERS):
    super(Model, self).__init__()
    self.vocab = vocab
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = n_layers
    tensor_embeddings = get_tensor_embeddings(input_size, EMBEDDING_DIM, self.vocab)
    self.encoder = nn.Embedding.from_pretrained(tensor_embeddings, freeze=False)
    self.gru = nn.GRU(EMBEDDING_DIM, hidden_size, n_layers)
    self.decoder = nn.Linear(hidden_size, output_size)
    
  def forward(self, inputchar, hidden):
    inputchar = self.encoder(inputchar.view(1, -1))
    output, hidden = self.gru(inputchar.view(1, 1, -1), hidden)
    output = self.decoder(output.view(1, -1))
    return output, hidden

  def init_hidden(self):
    if torch.cuda.is_available():
      return Variable(torch.zeros(self.n_layers, 1, self.hidden_size).cuda('cuda'))
    else:
      return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))

  def train_step(self, inp, criterion, optim,):
    self.zero_grad()
    loss = 0
    total_items = 0
    for i, line in enumerate(inp):
      if (i+1) % (BATCH_SIZE // 10) == 0:
        print('|', end='')
      elif (i+1) % (BATCH_SIZE // 100) == 0:
        print('-', end='')
      hidden = self.init_hidden()
      total_items += len(line) - 1
      for c in range(len(line) - 1):
        output, hidden = self(line[c], hidden)
        target = line[c + 1]
        # unsqueeze() is used to add dimension to the tensor
        loss += criterion(output, target.unsqueeze(dim=0))
    # Back propagation
    print("\nUpdating weights")
    loss.backward()
    optim.step()
    return loss.item() / total_items
    
  def predict(self, history='A'):
    self.eval()
    hidden = self.init_hidden()
    history_input = char_tensor(history)

    # Use priming string to "build up" hidden state
    for c in range(len(history) - 1):
      _, hidden = self(history_input[c], hidden)
    inp = history_input[-1]

    output, hidden = self(inp, hidden)
    # print(output, type(output))
    top_i = []
    for i in torch.argsort(output[0])[-3:]:
      top_i.append(i.item())
    
    # Add predicted character to string and use as next input
    predicted_chars = []
    for i in top_i:
      predicted_chars.append(self.vocab[i])
    self.train()
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

def run_train(model, train_data, work_dir):
  model.to(DEVICE)
  for param in model.parameters():
    if not param.is_cuda:
      print("Model not initialized as cuda")
      break

  optim = torch.optim.Adam(model.parameters(), lr=0.0005)
  criterion = nn.CrossEntropyLoss()
  start = time.time()

  eps = 250
  with open(os.path.join(work_dir, 'train_log.txt'), 'w') as f:
    print("Training model")
    for epoch in range(1, eps + 1):
      print(f"Epoch {epoch}")
      inp = train_data.random_training_set()
      loss = model.train_step(inp, criterion, optim)
      accuracy = (model.evaluate() * 100)
      epoch_summary = '[%s (%d %d%%) %.4f], Accuracy: %.3f%%' % (time_since(start), epoch, epoch / eps * 100, loss, accuracy)
      print(epoch_summary)
      f.write(epoch_summary)
      f.write('\n')
          
  with open('model.checkpoint.pt', 'wb') as f:
    torch.save(model, f)

  for history, next_char in test_data:
    preds = model.predict(history)
    print(next_char, preds, history)