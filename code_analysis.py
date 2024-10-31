import torch
import numpy as np
import argparse
from models import Code_Learner
# import torchtext.legacy as torchtext
import torchtext


import torch.nn as nn
import os

import utils

#This code is for evaluating acc/loss of models

# Use the GPU if it's available
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='Generates codes for words')
# Directory where we want to write everything we save in this script to
parser.add_argument('--M', default=utils.defaultM, type=int, metavar='N', help='Number of source dictionaries, default: 64')
parser.add_argument('--K', default=utils.defaultK, type=int, metavar='N', help='Source dictionary size, default: 8')
parser.add_argument('--words', default=['dog', 'cat'], metavar='[word1 ,word2]',help='words to find codes for')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='Minibatch size, default: 128')
parser.add_argument('--modelname', default=utils.modelname, metavar='MODEL_TYPE', help='Model name; select dataset. glove.6B.50d.txt or glove.42B.300d.txt, default: glove.42B.300d.txt')
parser.add_argument('--mode', default='iterative', metavar='MODEL_TYPE', help='Mode; iterative if automatic, else we use single M and K tuple, default: iterative')


#Method comparing codes from two words
def compare_codes(model, glove_dict, word1, word2):
    # Pass GloVe vectors into encoders to get codes
    vec1, vec2 = glove_dict[word1], glove_dict[word2]
    if use_gpu:
        vec1, vec2 = vec1.cuda(), vec2.cuda()
    _, code1 = model.encoder(vec1, training=False)
    _, code2 = model.encoder(vec2, training=False)
    print(word1, code1)
    print(word2, code2)
    return code1, code2

#Method outputs codes from a word
def compare_codes2(model, glove_dict, word1):
    # Pass GloVe vectors into encoders to get codes
    vec1 = glove_dict[word1]
    if use_gpu:
        vec1 = vec1.cuda()
    _, code1 = model.encoder(vec1, training=False)
    print(word1, code1)
    print("origianl vector:", vec1)

    codet = torch.transpose(code1, 0,1).float()
    comemb=model.decoder.forward(codet)
    print("compressed vector:", comemb)
    comemb=model(vec1, training=False)
    print("compressed vector:", comemb)
    return code1

#Method loading model and comparing codes.
def main(modelname=utils.modelname, sizeofM=utils.defaultM, sizeofK=utils.defaultK):
    global args
    # Parse commands from ArgumentParser
    args = parser.parse_args()
    # Load GloVE embeddings
    orig_embeddings = torch.load(utils.datapath(modelname) + 'all_orig_emb.pt')
    # Load all GloVE words
    with open(utils.datapath(modelname) + "glove_words.txt", "r") as file:
        glove_words = file.read().split('\n')
    # Recreate GloVE_dict
    glove_dict = {}
    for i, word in enumerate(glove_words):
        glove_dict[word] = orig_embeddings[i]
    # Load up the Code Learner model
    model = Code_Learner(utils.numfeature(modelname), sizeofM, sizeofK)
    try: 
        model = torch.load(utils.compressionpath(modelname,sizeofM, sizeofK, 'final'))
    except:
        model = torch.load(utils.compressionpath(modelname,sizeofM, sizeofK, 'final'), map_location=torch.device('cpu'))
    if use_gpu:
        model = model.cuda()

    model.train(False)

    # Generate codes
    #compare_codes(model, glove_dict, *args.words)
    compare_codes2(model, glove_dict, 'the')

#Evaluate Acc of Classifier
def ClassifierEvaluator(modelname, classifierpath):
    global args
    # Parse commands from ArgumentParser
    args = parser.parse_args()
    # Our text field for imdb data
    TEXT = torchtext.data.Field(lower=True)
    # Our label field for imdb data
    LABEL = torchtext.data.Field(sequential=False)
    # Load GloVE embeddings
    orig_embeddings = torch.load(utils.datapath(modelname) + 'all_orig_emb.pt')
    
    # Load shared words and all GloVE words
    with open(utils.datapath(modelname) + "shared_words.txt", "r") as file:
        shared_words = file.read().split('\n')
    with open(utils.datapath(modelname) + "glove_words.txt", "r") as file:
        glove_words = file.read().split('\n')
    # Recreate GloVE_dict
    glove_dict = {}
    for i, word in enumerate(glove_words):
        glove_dict[word] = orig_embeddings[i]

    # Load IMDB dataset with standard splits and restrictions identical to paper
    train, test = torchtext.datasets.IMDB.splits(TEXT, LABEL, filter_pred=lambda ex: ex.label != 'neutral' and len(ex.text) <= 400)

    # Both loops go through the words of train and test dataset, finds words without glove vectors, and replaces them with <unk>
    for i in range(len(train)):
        review = train.examples[i].text
        for i, word in enumerate(review):
            if word not in glove_dict:
                review[i] = '<unk>'
    for i in range(len(test)):
        review = test.examples[i].text
        for i, word in enumerate(review):
            if word not in glove_dict:
                review[i] = '<unk>'

    # Build modified vocabulary
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    # Create iterators over train and test set
    train_iter, test_iter = torchtext.data.BucketIterator.splits((train, test), batch_size=args.batch_size, repeat=False, device=-1)

    ########

    classifier = torch.load(classifierpath)
    try: 
        classifier = torch.load(classifierpath)
    except:
        classifier = torch.load(classifierpath, map_location=torch.device('cpu'))
    classifier.train(False)

    loss_func = nn.NLLLoss()

    ######   

    # VALIDATION
    valid_loss, valid_acc = utils.test_model(classifier, test_iter,loss_func,device)
        
    # Calculate accuracy and report
    print('''Classifier Loss: {l_v:.3f} Accuracy: {r_v:.3f}'''.format(l_v = valid_loss, r_v = valid_acc))


#Evaluate Acc of Classifier
def ClassifierEvaluator_logreg_sigmoid_checker(modelname, classifierpath):
    global args
    # Parse commands from ArgumentParser
    args = parser.parse_args()
    # Our text field for imdb data
    TEXT = torchtext.data.Field(lower=True)
    # Our label field for imdb data
    LABEL = torchtext.data.Field(sequential=False)
    # Load GloVE embeddings
    orig_embeddings = torch.load(utils.datapath(modelname) + 'all_orig_emb.pt')
    
    # Load shared words and all GloVE words
    with open(utils.datapath(modelname) + "shared_words.txt", "r") as file:
        shared_words = file.read().split('\n')
    with open(utils.datapath(modelname) + "glove_words.txt", "r") as file:
        glove_words = file.read().split('\n')
    # Recreate GloVE_dict
    glove_dict = {}
    for i, word in enumerate(glove_words):
        glove_dict[word] = orig_embeddings[i]

    # Load IMDB dataset with standard splits and restrictions identical to paper
    train, test = torchtext.datasets.IMDB.splits(TEXT, LABEL, filter_pred=lambda ex: ex.label != 'neutral' and len(ex.text) <= 400)

    # Both loops go through the words of train and test dataset, finds words without glove vectors, and replaces them with <unk>
    for i in range(len(train)):
        review = train.examples[i].text
        for i, word in enumerate(review):
            if word not in glove_dict:
                review[i] = '<unk>'
    for i in range(len(test)):
        review = test.examples[i].text
        for i, word in enumerate(review):
            if word not in glove_dict:
                review[i] = '<unk>'

    # Build modified vocabulary
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    # Create iterators over train and test set
    train_iter, test_iter = torchtext.data.BucketIterator.splits((train, test), batch_size=args.batch_size, repeat=False, device=-1)

    ########

    classifier = torch.load(classifierpath)
    try: 
        classifier = torch.load(classifierpath)
    except:
        classifier = torch.load(classifierpath, map_location=torch.device('cpu'))
    classifier.train(False)

    loss_func = nn.BCELoss()

    ######   

    # VALIDATION
    avg_val = utils.test_model_logreg_analysis(classifier, test_iter,loss_func,device)
    print(avg_val)
    # valid_loss, valid_acc = utils.test_model(classifier, test_iter,loss_func,device)
        
    # Calculate accuracy and report
    # print('''Classifier Loss: {l_v:.3f} Accuracy: {r_v:.3f}'''.format(l_v = valid_loss, r_v = valid_acc))



#Evaluate loss of Compression
def CompressionEvaluator(modelname, compressionpath):
    global args
    # Parse commands from ArgumentParser
    args = parser.parse_args()
    # Load a model
    try:
        model = torch.load(compressionpath)
    except:
        model = torch.load(compressionpath, map_location=torch.device('cpu'))
    model.train(False)

    # Put model into CUDA memory if using GPU
    if use_gpu:
        model.cuda()
    # Load GloVE baseline embeddings
    orig_embeddings = torch.load(utils.datapath(modelname) + 'all_orig_emb.pt')

    batch_size=args.batch_size
    loss_func = nn.MSELoss(size_average=True)

    #####
    
    # Define total number of words with GloVE vectors
    total_words = len(orig_embeddings)
    # Assign 90% of the words to be part of the training dataset
    train_len = int(len(orig_embeddings) * 0.9)
    train_indices = np.random.choice(total_words, train_len)
    # Let the remaining indices be part of a validation set
    valid_indices = np.array(list(set(range(total_words)) - set(train_indices)))

    # VALIDATION
    
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

    print('''Compression Validation_Recon_Loss: {r_l:.3f}'''.format(r_l = valid_loss))


#Iteratively evaluate classifier/compression
def iterative():
    KandM=utils.KandM
    global args
    # Parse commands from ArgumentParser
    args = parser.parse_args()

    for modelname in utils.models_glove:
        for key, value in KandM.items():
            for m in value:    
                print('''M and K: {name}'''.format(name = utils.dataname(modelname,m, key)))
                localclassifierpath= utils.classifierpath(modelname, m, key, 'final')
                ClassifierEvaluator(modelname, localclassifierpath)
                # localclassifierpath= utils.classifierpath(modelname, m, key, tag = '_concat')
                # ClassifierEvaluator_logreg_sigmoid_checker(modelname, localclassifierpath)
                if key*m!=0:
                    localcompressionpath=utils.compressionpath(modelname, m, key, 'final')
                    CompressionEvaluator(modelname, localcompressionpath)
                    #main(args.modelname, localcompressionpath)



if __name__ == '__main__':

    global args
    args = parser.parse_args()

    #main(args.modelname, 8,8)

    if args.mode=='iterative':
        iterative()
    if args.mode=='main':
        localcompressionpath=utils.compressionpath(args.modelname,args.M, args.K, 'final')
        main(args.modelname, args.M, args.K)
    if args.mode=='CompressionEvaluator':
        localcompressionpath=utils.compressionpath(args.modelname,args.M, args.K, 'final')
        CompressionEvaluator(args.modelname, localcompressionpath)
    if args.mode=='indicesToTxt':
        localdatafolder = utils.dataname(args.modelname, args.M, args.K)
        localclassifierpath= utils.classifierpath(args.modelname, args.M, args.K, 'final')
        ClassifierEvaluator(args.modelname, localclassifierpath)
