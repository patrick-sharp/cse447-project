import string
import os
import random
import unidecode
import numpy as np

CHUNK_LEN = 200
#This is the length of the pretrained embeddings being used
EMBEDDING_DIM = 300
DATA_DIR = "../data" # Where the training data is stored
EMBEDDINGS_PATH = "./embeddings.txt"
all_characters = string.printable


#Update to random_chunk function to get a random chunk from a random file
def random_chunk():
    #generates a list of files in the training directory and chooses one randomly
    fileList = os.listdir(DATA_DIR)
    fileName = fileList[random.randint(0, len(fileList) -1)]
    fileRaw = open(training_dir + fileName, 'r')
    
    #reads the file into a string and then chooses a random chunk of that string to return
    file = unidecode.unidecode(fileRaw.read())
    start_index = random.randint(0, len(file) - chunk_len)
    end_index = start_index + chunk_len + 1
    ret = file[start_index:end_index]
    fileRaw.close()
    return ret

#Gets pretrained character embeddings from the path given at the top of the file
#Returns a np matrix in the order of string.printable with all the given embeddings
#Matrix has random embeddings if the given file didn't have an embedding for a charater
def get_embeddings():
    embeddings = {}
    file = open(EMBEDDINGS_PATH, 'r')
    for line in file:
        raw = line.strip().split()
        #First value in the line is the character name the rest are float values
        embedValues = np.asarray(raw[1:], dtype=float)
        char = raw[0]

        embeddings[char] = embedValues

    #Empty matrix going to be ordered by string.printable
    orderedWeights = np.zeros((len(all_characters), EMBEDDING_DIM))
    
    for i, char in enumerate(all_characters):
        try:
            #Uses the given embedding if present
            orderedWeights[i] = embeddings[char]
        except KeyError:
            #Generates a random embedding if not
            orderedWeights[i] = np.random.normal(scale=0.6, size=embedding_dim)

    return orderedWeights





