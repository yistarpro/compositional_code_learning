import torch
# import torchtext.legacy as torchtext
import os
import numpy as np
import argparse
import torch.nn as nn
from models import Code_Learner, Classifier

from torchtext import data
from torchtext import datasets

import random
import utils

#4. Save parameters from models

# Use the GPU if it's available
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='Generates codes for words')
# Directory where we want to write everything we save in this script to
parser.add_argument('--M', default=utils.defaultM, type=int, metavar='N', help='Number of source dictionaries, default: 64')
parser.add_argument('--K', default=utils.defaultK, type=int, metavar='N', help='Source dictionary size, default: 8')
parser.add_argument('--modelname', default=utils.modelname, metavar='MODEL_TYPE', help='Model name; select dataset. glove.6B.50d.txt or glove.42B.300d.txt, default: glove.42B.300d.txt')
parser.add_argument('--mode', default='iterative', metavar='MODEL_TYPE', help='Mode; iterative if automatic, else we use single M and K tuple, default: iterative')


#word, code, embedding
#code to vector

#Save tuple of word and codes
def indicesToTxt(modelname, datafolder, compressionpath, sizeofM, sizeofK):
    # Load a model
    model = Code_Learner(utils.numfeature(modelname), sizeofM, sizeofK)
    try: 
        model = torch.load(compressionpath)
    except:
        model = torch.load(compressionpath, map_location=torch.device('cpu'))
    # Put model into CUDA memory if using GPU
    if use_gpu:
        model.cuda()
    # Load GloVE baseline embeddings
    orig_embeddings = torch.load(utils.datapath(modelname) + 'all_orig_emb.pt')
    print(len(orig_embeddings))
    # Load all GloVE words
    with open(utils.datapath(modelname) + "glove_words.txt", "r") as file:
        glove_words = file.read().split('\n')
    # Recreate GloVE_dict
    glove_dict = {}
    for i, word in enumerate(glove_words):
            glove_dict[word] = orig_embeddings[i]

    indices = {}

    #extracting codes
    for i, word in enumerate(glove_words):
        vec = glove_dict[word]
        if use_gpu:
            vec = vec.cuda()
        _, code = model.encoder(vec, training=False)

        cdlst=[]
        for j in code.cpu().detach().numpy()[0]:
            cdlst.append(str(j))
        indices[word]=cdlst

    # We write our glove_word list to a text file for easy reference
    with open(utils.loc+'models/'+datafolder+'/'+utils.dataname(modelname, sizeofM, sizeofK) +'wordtoindex.txt', 'w') as out_file:
        #out_file.write('\n'.join(weighttext))
        for key, value in indices.items():  
            out_file.write('%s, %s\n' % (key, value))
    print('''wordtoindex generated with {name}'''.format(name = datafolder))

#Save compressed embedding result to txt
def embedToTxt(modelname = utils.modelname):
    datapath=utils.datapath(modelname)

    # Load GloVE baseline embeddings
    orig_embeddings = torch.load(datapath + 'all_orig_emb.pt')

    # Load all GloVE words
    with open(datapath + "glove_words.txt", "r") as file:
        glove_words = file.read().split('\n')
    # Recreate GloVE_dict
    glove_dict = {}
    for i, word in enumerate(glove_words):
        glove_dict[word] = orig_embeddings[i]

    indtoemb = {}

    #extracting codes
    for i, word in enumerate(glove_words):
        lst=[]
        for elmt in glove_dict[word].cpu().detach().numpy():
            lst.append(str(elmt))
        indtoemb[word]=lst

    with open(datapath +'indextoemb.txt', 'w') as out_file:
        #out_file.write('\n'.join(weighttext))
        for key, value in indtoemb.items():  
            out_file.write('%s, %s\n' % (key, value))

#Save codebooks of model
def codebooks(modelname, datafolder, compressionpath, sizeofM, sizeofK):

    # Load a model
    model = Code_Learner(utils.numfeature(modelname), sizeofM, sizeofK)
    try: 
        model = torch.load(compressionpath)
    except:
        model = torch.load(compressionpath, map_location=torch.device('cpu'))
    if use_gpu:
        model = model.cuda()
    weights = model.decoder.A.weight
    weighttext={}
    #extracting codes
    for i, codeb in enumerate(weights):
        codebt = torch.transpose(codeb, 0,1)
        for j, cd in enumerate(codebt):
            ind=i*sizeofK+j
            lst=[]
            for elmt in cd.cpu().detach().numpy():
                lst.append(str(elmt))

            weighttext[str(ind)]=lst

    # We write our glove_word list to a text file for easy reference
    with open(utils.loc+'models/'+datafolder+'/'+utils.dataname(modelname, sizeofM, sizeofK) +'weight.txt', 'w') as out_file:
        #out_file.write('\n'.join(weighttext))
        for key, value in weighttext.items():  
            out_file.write('%s, %s\n' % (key, value))
    
    print('''codebook generated with {name}'''.format(name = datafolder))


def sentences(numofpairs = 1024):
    # Our text field for imdb data
    TEXT = data.Field(lower=True)
    # Our label field for imdb data
    LABEL = data.Field(sequential=False)

    with open('./data6B50d/' + "glove_words.txt", "r") as file:
        glove_words = file.read().split('\n')

    # Load IMDB dataset with standard splits and restrictions identical to paper
    train, test = datasets.IMDB.splits(TEXT, LABEL, filter_pred=lambda ex: ex.label != 'neutral' and len(ex.text) <= 400)

    if not os.path.exists(utils.loc+'models/sentences'):
        os.makedirs(utils.loc+'models/sentences')

    indices = random.sample(range(len(test)), numofpairs)

    with open( utils.loc+'models/sentences/sentences.txt', 'w') as out_file:
        #out_file.write('\n'.join(weighttext))
        for i in indices:
            review = test.examples[i].text
            for j, word in enumerate(review):
                if word not in glove_words:
                    review[j] = '<unk>'
            if test.examples[i].label == 'pos':
                lab = 1
            else:
                lab = 0
            out_file.write('%s, %s\n' % (lab, review))

    print('Sentence set saved')


def logregweights(modelname, datafolder, sizeofM, sizeofK):
    model = torch.load(utils.classifierpath(modelname,sizeofM, sizeofK, tag='_logreg', check='final'))

    lst = []
    for i in list(model.parameters())[1][0].cpu().detach().numpy():
        lst.append(str(i))
    lst.append(str(list(model.parameters())[2][0].cpu().detach().numpy()))

    dirtmp = utils.classifierpath(modelname,sizeofM, sizeofK, tag='_logreg', check='final')
    opath = dirtmp[:-len(dirtmp.split('/')[-1])]+utils.dataname(modelname, sizeofM, sizeofK)+'logreg.txt'

    with open(opath, 'w') as out_file:
    # with open(utils.loc+'models/'+datafolder+'/'+utils.dataname(modelname, sizeofM, sizeofK) +'logreg.txt', 'w') as out_file:
        #out_file.write('\n'.join(weighttext))
        out_file.write('%s' % (lst))
    
    print('''logreg weight generated with {name}'''.format(name = datafolder))



#Iteratively save all
def iterative():
    localMandK=utils.KandM

    for modelname in utils.models_glove:
        for key, value in localMandK.items():
            for m in value:
                if m!=0:
                    print('''M and K: {name}'''.format(name = utils.dataname(modelname,m, key)))
                    localdatafolder = utils.dataname(modelname, m, key)
                    localcompressionpath=utils.compressionpath(modelname, m, key)
                    logregweights(modelname, localdatafolder, m, key)
                    indicesToTxt(modelname, localdatafolder, localcompressionpath, m, key)
                    codebooks(modelname, localdatafolder, localcompressionpath, m, key)

if __name__ == '__main__':
    global args
    args = parser.parse_args()
    if args.mode=='iterative':
        sentences(1024)
        iterative()
    if args.mode=='codebooks':
        localdatafolder = utils.dataname(args.modelname, args.M, args.K)
        localcompressionpath=utils.compressionpath(args.modelname,args.M, args.K)
        codebooks(args.modelname, localdatafolder, localcompressionpath, args.M, args.K)
    if args.mode=='embedToTxt':
        localdatafolder = utils.dataname(args.modelname, args.M, args.K)
        localcompressionpath=utils.compressionpath(args.modelname,args.M, args.K)
        embedToTxt(args.modelname)
    if args.mode=='indicesToTxt':
        localdatafolder = utils.dataname(args.modelname, args.M, args.K)
        localcompressionpath=utils.compressionpath(args.modelname,args.M, args.K)
        indicesToTxt(args.modelname, localdatafolder, localcompressionpath, args.M, args.K)
    if args.mode=='logregweights':
        localdatafolder = utils.dataname(args.modelname, args.M, args.K)
        localcompressionpath=utils.compressionpath(args.modelname,args.M, args.K)
        logregweights(args.modelname, localdatafolder, args.M, args.K)
