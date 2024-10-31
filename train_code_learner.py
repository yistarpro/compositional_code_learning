import torchtext
import os
import torch.optim as optim
import torch
import numpy as np
import math
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models import Code_Learner
import time
import utils

#2. Word embedding compression.
#This file is not modified much, changed structure for variable inputs.

# Use the GPU if it's available
use_gpu = torch.cuda.is_available()
parser = argparse.ArgumentParser(description='Train Code_Learner to encode words and create embeddings')
# Directory where we want to write everything we save in this script to
parser.add_argument('--models_folder', default=utils.loc+'models/', metavar='DIR',
                    help='folder to save models')
parser.add_argument('--embedding_size', default=utils.numfeature(), type=int, metavar='N', help='Embedding dimension size, default: 300')
parser.add_argument('--M', default=utils.defaultM, type=int, metavar='N', help='Number of source dictionaries, default: 64')
parser.add_argument('--K', default=utils.defaultK, type=int, metavar='N', help='Source dictionary size, default: 8')
parser.add_argument('--lr', default=0.0001, type=float, metavar='N', help='Adam learning rate, default: 0.0001')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='Minibatch size, default: 128')
parser.add_argument('--epochs', default=200000, type=int, metavar='N', help='Total number of epochs, default: 200,000')
parser.add_argument('--modelname', default=utils.modelname, metavar='MODEL_TYPE', help='Model name; select dataset. glove.6B.50d.txt or glove.42B.300d.txt, default: glove.42B.300d.txt')
parser.add_argument('--mode', default='iterative', metavar='MODEL_TYPE', help='Mode; iterative if automatic, else we use single M and K tuple, default: iterative')


def train(epochs, last_epoch, batch_size, model, optimizer, loss_func, orig_embeddings, dataname, tau=1):
    model.train()
    # Initialize validation loss for model evaluation
    best_val_loss = float('inf')
    # Define total number of words with GloVE vectors
    total_words = len(orig_embeddings)
    # Assign 90% of the words to be part of the training dataset
    train_len = int(len(orig_embeddings) * 0.9)
    train_indices = np.random.choice(total_words, train_len)
    # Let the remaining indices be part of a validation set
    valid_indices = np.array(list(set(range(total_words)) - set(train_indices)))

    # TRAINING PROCESS- each epoch is really a minibatch sample
    for epoch in range(epochs):
        model.train()
        # Let our training batch be a random selection of batch_size from the training set
        train_batch = orig_embeddings[np.random.choice(train_indices, batch_size)]
        # Because our model is an autoencoder, the data = target
        data, target = train_batch, train_batch
        # If using GPU, put into CUDA memory
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        # Clears gradients of tensors
        optimizer.zero_grad()
        # Computes encoding embedding
        comp_emb = model(data, tau)
        # Calculate reconstruction loss
        loss = loss_func(comp_emb, target)
        # Compute sum of gradients
        loss.backward()
        # Perform optimization step
        optimizer.step()

        # VALIDATION
        # Every 1000 iterations, check our model with a validation set
        if epoch % 500 == 0:
            model.eval()
            # Let our validation batch be a random selection of batch_size from the validation set
            valid_batch = orig_embeddings[np.random.choice(valid_indices, batch_size)]
            # Because our model is an autoencoder, the data = target
            data, target = valid_batch, valid_batch
            # If using GPU, put into CUDA memory
            if use_gpu:
                data, target = data.cuda(), target.cuda()
            # Computes encoding embedding
            comp_emb = model(data)
            # Calculate reconstruction loss
            loss = loss_func(comp_emb, target)
            valid_loss = loss.item()
            # Save model if this is our lowest validation loss
            if valid_loss < best_val_loss:
                torch.save(model, args.models_folder + dataname + '/epoch_' + str(epoch+last_epoch+1) + '.pt')
                best_val_loss = valid_loss
            if epoch+1 == epochs:
                torch.save(model, args.models_folder + dataname + '/finalepoch_' + str(epoch+last_epoch+1) + '.pt')
            # Every 10,000 iterations, print the reconstruction loss
            if epoch % 500 == 0:
                print('''Epoch [{e}/{num_e}]\t Validation_Recon_Loss: {r_l:.3f}'''.format(e=epoch+1+last_epoch, num_e=epochs+last_epoch, r_l = valid_loss))

def main(modelname=utils.modelname, sizeofM=utils.defaultM, sizeofK=utils.defaultK):
    global args
    dataname=utils.dataname(modelname, sizeofM, sizeofK)
  
    # Parse commands from ArgumentParser
    args = parser.parse_args()
    # If our particular model's folder doesn't exist, create it
    if not os.path.exists(args.models_folder + dataname + '/'):
        os.makedirs(args.models_folder + dataname + '/')

    last_epoch=utils.compressionepoch(modelname, sizeofM, sizeofK)

    # Create a model
    model = Code_Learner(utils.numfeature(modelname), sizeofM, sizeofK)
    if last_epoch != 0:
        try:
            model = torch.load(utils.compressionpath(modelname, sizeofM, sizeofK))
        except:
            model = torch.load(utils.compressionpath(modelname, sizeofM, sizeofK), map_location=torch.device('cpu'))

    # Initialize an optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Define the loss function
    loss_func = nn.MSELoss(size_average=True)
    # Put model into CUDA memory if using GPU
    if use_gpu:
        model.cuda()
    # Load GloVE baseline embeddings
    orig_embeddings = torch.load(utils.datapath(modelname) + 'all_orig_emb.pt')
    # Train and save the model
    st = time.time()
    train(args.epochs-last_epoch, last_epoch, args.batch_size, model, optimizer,loss_func, orig_embeddings, dataname)
    et = time.time()
    print("Elapsed time for training: %s"%(et-st))


def iterative():
    KandM=utils.KandM
    
    for modelname in utils.models_full:
        for key, value in KandM.items():
            for m in value:
                if m*key!=0:
                    #os.abort
                    print('''M and K: {name}'''.format(name = utils.dataname(modelname, m, key)))
                    main(modelname, m, key)


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    if args.mode=='iterative':
        iterative()
    else:
        main(args.modelname, args.M, args.K)

