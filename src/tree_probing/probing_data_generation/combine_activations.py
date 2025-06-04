# import json, h5py
import numpy as np
import argparse
import pickle
import torch
from tqdm import tqdm

def combine_activations_np(sentarray, mode='avg', round_to=-1):
    if mode in ['avg', 'max']:
        combined_arr = torch.zeros((sentarray.shape[0] - 1, sentarray.shape[1]))

    elif mode == 'concat':
        combined_arr = torch.zeros((sentarray.shape[0] - 1, sentarray.shape[1]*2))

    for i in range(sentarray.shape[0] - 1):
        i_repr = sentarray[i,:][None,:]
        j_repr = sentarray[i+1,:][None,:]

        if mode == 'avg':
            avg_i_j = torch.mean(torch.stack([i_repr, j_repr]), dim=0)

            if round_to > -1:
                avg_i_j = torch.round(avg_i_j, decimals=round_to)
            combined_arr[i,:] = avg_i_j

        elif mode == 'concat':
            concat_i_j = torch.cat([i_repr, j_repr], dim=1)
            if round_to > -1:
                concat_i_j = torch.round(concat_i_j, decimals=round_to)
            combined_arr[i,:] = concat_i_j

        elif mode == 'max':
            stacked = torch.stack([i_repr,j_repr])
            combined_arr[i,:] = torch.max(torch.abs(stacked), dim=0)[0] # max returns a tuple (max, argmax)
        else:
            raise NotImplementedError 
    
    return combined_arr 

def extract_from_pickle(infilename, outfilename, reltoksfilename, mode='avg', round_to=-1):
    # load pickle
    with open(infilename, 'rb') as f:
        sent_activations = pickle.load(f)

    with open(reltoksfilename, 'r') as f:
        rel_toks_sents = f.readlines()
    
    rel_toks_sents = [x.strip().split() for x in rel_toks_sents]

    output_dict = {}
    for (layer, activations) in tqdm(sent_activations.items()):
        output_dict[layer] = []

        for i, sent_act in enumerate(tqdm(activations, leave=False)): 
            output_act = combine_activations_np(sent_act, mode=mode, round_to=round_to)
            output_dict[layer].append(output_act)
            assert len(rel_toks_sents[i]) == output_act.shape[0], f"Length of sentence {len(rel_toks_sents[i])} ({i}) does not match length of activations {output_act.shape[0]}"

    # save pickle
    with open(outfilename, 'wb') as f:
        pickle.dump(output_dict, f)

def concatenate_layers(infilename, outfilename):
    """
    Concatenates the activations from layers 3, 6 and 8
    """
    # read pickle file
    with open(infilename, 'rb') as f:
        sent_activations = pickle.load(f)

    output = {}
    output[0] = []
    for l3, l6, l8 in zip(sent_activations[3], sent_activations[6], sent_activations[8]):
        output[0].append(torch.cat([l3, l6, l8], dim=1))
    
    # save pickle file
    with open(outfilename, 'wb') as f:
        pickle.dump(output, f)

if __name__=='__main__':
    """
    Take a file with LM activations and create a new file with averaged representation:
    For a sentence with length n, a new artificial sentence is created with length sum(0,...,n)
    The first token in that new sentence is the average of tokens 0 and 0 in the original, the second the average
    of tokens 0 and 1, and so on (for the whole sentence)

    -rel_toks txt file is used for the sentence_to_index dictionary  
    -m <avg or concat or max or left or right> optional: specify if the activations should be averaged or concatenated or if the max should be taken (using absolute values). Default is average. Only works for hdf5, not json
    -round <int> optional: round activation values to so many decimals (default: do not round at all). Only works for hdf5, not json
    -sampled: use this option if representations should only be combined for a sampled portion of the data. Only works for hdf5, not json
    -concatenate_layers: use this option if you want to concatenate the layers 3 6 and 9. Made for activations_combined_concat.pickle

    usage: 
    ...
    
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file')
    parser.add_argument('--output_file')
    parser.add_argument('--format', default='pickle')
    parser.add_argument('--rel_toks', required=True)
    parser.add_argument('--mode', default='avg', choices=['avg', 'concat', 'max', 'left', 'right'],
                        help='avg: average, concat: concatenate, max: take max, left: only use left token, right: only use right token')
    parser.add_argument('--round_to', default=-1, type=int)
    parser.add_argument('--concatenate_layers', action=argparse.BooleanOptionalAction,
                        help='Use this function if you want to concatenate the layers 3 6 and 9')
    parsedargs = parser.parse_args()
    
    if parsedargs.concatenate_layers:
        concatenate_layers(parsedargs.input_file, parsedargs.output_file)
    else:
        extract_from_pickle(parsedargs.input_file, parsedargs.output_file, parsedargs.rel_toks, mode=parsedargs.mode, round_to=parsedargs.round_to)