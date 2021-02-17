import string
import random
import os

import torch

from Model import Model

class ModelRunner:
    """
    This is our project's main class.
    """
    def __init__(self, model=Model()):
        self.model = model

    @classmethod
    def load_training_data(cls):
        # your code here
        # this particular model doesn't train
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
        # your code here
        pass

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
