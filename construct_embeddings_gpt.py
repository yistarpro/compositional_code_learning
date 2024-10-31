import torchtext
import torch
import numpy as np
import argparse
from torchtext.legacy import data
from torchtext.legacy import datasets
import utils
from itertools import product

from transformers import GPT2Model, GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
embedding_layer = model.get_input_embeddings()
embedding_matrix = embedding_layer.weight.data

tokens = list(tokenizer.get_vocab().keys())

#1. Construct Original Embedding.
#This file is not modified much, just changed some variable names.

parser = argparse.ArgumentParser(description='Create GloVE and IMDB embeddings')
# File name for GloVE vectors
parser.add_argument('--glove_file', default=utils.modelname, metavar='file',
                    help='file which contains GloVE embeddings')
# Directory where we want to write everything we save in this script to
parser.add_argument('--data_folder', default=utils.datapath(utils.modelname), metavar='DIR',
                    help='folder to save embeddings, data, text files, etc.')

def main():

    global args
    # Parse commands from ArgumentParser
    args = parser.parse_args()    
    modelname=args.glove_file

    # Our text field for imdb data
    # TEXT = torchtext.data.Field(lower=True, fix_length=400)
    TEXT = data.Field(lower=True, fix_length=400)
    # Our label field for imdb data
    # LABEL = torchtext.data.Field(sequential=False)
    LABEL = data.Field(sequential=False)

    # Use standard split for IMDB dataset, filtering out reviews that are longer than 400 words
    train, test = datasets.IMDB.splits(TEXT, LABEL, \
    filter_pred=lambda ex: ex.label != 'neutral' and len(ex.text) <= 400)
    # Build vocabulary from training dataset
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    # Get a list of all words from imdb
    imdb_words = TEXT.vocab.freqs.keys()

    # Next, construct a dictionary that maps words to their GloVe vectors
    glove_dict = {}
    # We also want a list of all glove_words, because it is handy
    glove_words = []
    # Reading previously specified file
 
    # There are this many words included in the file
    total_glove_num = len(tokens)
    # We also want to store all the embeddings to a file, which we can't do from a dict
    all_orig_embeddings = torch.zeros(total_glove_num, utils.numfeature(modelname), dtype=torch.float)
    for i, embedding in enumerate(embedding_matrix):
        # Also to our FloatTensor for all GloVe embeddings
        all_orig_embeddings[i] = torch.FloatTensor(embedding)
    print('GloVe dict constructed')

    # Now we make a list of words that appear in both the IMDB dataset and the GloVE file
    shared_words = []
    for word in imdb_words:
        if word in tokens:
            shared_words.append(word)
    print('Shared words list constructed.')
    # We write our shared_word list to a text file for easy reference
    with open(utils.datapath(modelname) + 'shared_words.txt', 'w') as out_file:
        out_file.write('\n'.join(shared_words))

    # We write our glove_word list to a text file for easy reference
    with open(utils.datapath(modelname) + 'glove_words.txt', 'w') as out_file:
        out_file.write('\n'.join(tokens))

    # We save our glove_embedding for later use
    torch.save(all_orig_embeddings, utils.datapath(modelname) + 'all_orig_emb.pt')

if __name__ == '__main__':
    main()
