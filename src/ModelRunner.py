import string
import random
import os
import time
import math

import torch
import torch.nn as nn

from Model import Model
from data_handler import random_training_set

class ModelRunner:
    """
    This is our project's main class.
    """
    def __init__(self, model=Model()):
        self.model = model

    @classmethod
    def load_training_data(cls):
        # see get_data.sh for our code that downloads data
        return []

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        optim = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        criterion = nn.CrossEntropyLoss()
        start = time.time()
        all_losses = []
        loss_avg = 0

        eps = 1500
        for epoch in range(1, eps + 1):
            inp, target = random_training_set()
            loss = self.model.train(inp, target, criterion, optim)
            if True: #epoch % 60 == 0:
                print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / eps * 100, loss), end=' ')
                print("Accuracy: {}%".format(self.model.evaluate() * 100))

    def run_pred(self, data):
        preds = []
        for inp in data:
            # this model just predicts a random character each time
            top_guesses = self.model.predict(inp)
            preds.append(''.join(top_guesses))
        return preds
        return self.model.predict(data)

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint.pt'), 'wb') as f:
            torch.save(self.model, f)

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        try:
            with open(os.path.join(work_dir, 'model.checkpoint.pt'), 'rb') as f:
                model = torch.load(f)
            return ModelRunner(model=model)
        except:
            # if there is no saved model, just spin up a fresh one.
            # Note: this model will be completely untrained.
            print("No model checkpoint found. New model was initialized from scratch and NOT trained.")
            return ModelRunner(model=Model())

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)