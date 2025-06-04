from treetoolbox import lowest_phrase_above_leaf_i
# during test: from data_prep.treetoolbox
import sys
import argparse
import re
import numpy as np
import nltk
from tqdm import tqdm
from utils import load_tokenizer
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


punct_regex = re.compile(r"[^\w][^\w]?")

def findBIESTag(i: int, tree : nltk.Tree, i_tok: str, with_phrase_label=False) -> str:
    """
    Find BIES tag for token i in tree. If multiple tags, than the one
    with the lowest phrase label is chosen.
    Input:
        i: index of token
        tree: nltk.Tree
        i_tok: token at index i
    Output:
        tag: BIES tag for token i
    """
    # Check if token is punctuation
    if punct_regex.match(i_tok) and '<apostrophe>' not in i_tok:
        return 'PCT'
    
    phrase_label, phrase_node, ga_of_phrase_node = lowest_phrase_above_leaf_i(i, tree, return_target_ga=True)

    ga_of_leaf = tree.treeposition_spanning_leaves(i,i+1)
    ga_phrase_to_leaf = ga_of_leaf[len(ga_of_phrase_node):]

    is_beginning = ga_phrase_to_leaf[0] == 0 and (len(set(ga_phrase_to_leaf))==1)
    is_end = True
    node = phrase_node

    for k in ga_phrase_to_leaf:
        if len(node) - 1 > k:
            is_end=False
            break
        else:
            node = node[k]

    # Assign shortest BIES tag   
    if is_beginning and is_end:
        # Single-token phrase
        tag = 'S'
    elif is_beginning:
        # Beginning of phrase
        tag = 'B'
    elif is_end:
        # End of phrase
        tag = 'E'
    else:
        # Inside phrase
        tag = 'I'

    if with_phrase_label:
        if phrase_label.startswith('NP') and len(phrase_label.split('-')) > 1:
            tag+='-'+'-'.join(phrase_label.split('-')[:2])
        else:
            tag+='-'+phrase_label.split('-')[0]
    return tag

def biesLabels(tree, tokenizer, with_phrase_labels=False, skip_unkown_tokens=False):
    """
    Loop through through leaves in tree and assign BIES labels to each token
    Input:
        tree: nltk.Tree
    Output: 
        text_toks: list of tokens in tree
        bies_labels: list of BIES labels for each token in tree
    """
    sent = tree.leaves()
    text_toks = []
    bies_labels = []

    for i in range(len(sent)):
        i_tok = sent[i]

        # find phrase above token i
        if skip_unkown_tokens:
            if i_tok not in tokenizer.vocab:
                continue
        label = findBIESTag(i, tree, i_tok, with_phrase_label=with_phrase_labels)

        text_toks.append(i_tok)
        bies_labels.append(label)

    return text_toks, bies_labels

if __name__=='__main__':
    """
    Usage: 
    python3 span_prediction_format -ptb_tr <file_pattern> -ptb_notr <file_pattern> -text_toks <filename> -bies_labels <filename>

    Input Options: 
    - PTB with and without traces (both are needed)

    Output Options (each file has same number of lines): 
    - text_toks txt file with one sentence per line (input file for computing activations)
    - bies_labels txt file where the k-th label in the l-th line represents the beginning/inside/end/only label in the PTB of the corresponding tokens in text_toks.txt

    other options
    - cutoff x an integer x such that the scripts stops after processing x sentences
    -max_sent_length an integer x for the maximum sentence length (suggested: 20)
    
    The script asserts that all output files have the same number of lines, and that the output files text_toks and bies_labels have the same number of elements per line
    """

    ###############
    # PREP
    ###############

    """
    Call with: python extract_bies_labels.py -data /Users/sperdijk/Documents/Master/"Jaar_3"/Thesis/thesis_code/pcfg-lm/src/lm_training/corpora/eval_trees_10k.txt -text_toks /Users/sperdijk/Documents/Master/"Jaar_3"/Thesis/thesis_code/data/train_text_bies.txt -bies_labels /Users/sperdijk/Documents/Master/"Jaar_3"/Thesis/thesis_code/data/train_bies_labels.txt -max_sent_length 31
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default='/Users/sperdijk/Documents/Master/"Jaar_3"/Thesis/thesis_code/pcfg-lm/src/lm_training/corpora/eval_trees_10k.txt')
    parser.add_argument('--text_toks', type=Path, default='/Users/sperdijk/Documents/Master/"Jaar_3"/Thesis/thesis_code/data/train_text_bies.txt') 
    parser.add_argument('--bies_labels', type=Path, default=' /Users/sperdijk/Documents/Master/"Jaar_3"/Thesis/thesis_code/data/train_bies_labels.txt') 
    parser.add_argument('--cutoff')
    parser.add_argument('--duplicates', action='store_true')
    parser.add_argument('--model_id', type=str, default='babyberta')
    parser.add_argument('--max_sent_length', type=int, default=31)
    parser.add_argument('--with_phrase_labels', action='store_true')
    parsedargs = parser.parse_args()

    ignored_sents = []
    ignore_list = []

    if torch.cuda.is_available():
        # For running on snellius
        device = torch.device("cuda")
        print('Running on GPU.')
    # elif torch.backends.mps.is_available():
    #     # For running on M1
    #     device = torch.device("mps")
    #     print('Running on M1 GPU.')
    else:
        # For running on laptop
        device = torch.device("cpu")
        print('Running on CPU.')
    
    logger.info('Loading model...')
    tokenizer = load_tokenizer(parsedargs)
    logger.info('Model loaded.')

    # Used during development with a lower cutoff
    cutoff = np.inf
    if parsedargs.cutoff is not None:
        cutoff = int(parsedargs.cutoff)
    max_sent_length = np.inf
    if parsedargs.max_sent_length is not None:
        max_sent_length = int(parsedargs.max_sent_length)
    
    # Reading in input trees
    try:
        with open(parsedargs.data_dir) as f:
            tree_corpus = [nltk.Tree.fromstring(l.strip()) for l in f]
    except FileNotFoundError:
        logger.error(f"File not found: {parsedargs.data_dir}")
        sys.exit(1)

    parsedargs.text_toks.parent.mkdir(parents=True, exist_ok=True)
    parsedargs.bies_labels.parent.mkdir(parents=True, exist_ok=True)

    text_toks_file = open(parsedargs.text_toks, 'w')
    bies_labels_file = open(parsedargs.bies_labels, 'w')

    binary = False
    label_counts = dict()

    output_sents = set()
    ###############
    # Conversion 
    ###############
    for k, tree in enumerate(tqdm(tree_corpus)):
        if k - len(ignored_sents) > cutoff:
            continue
        # some sentences are not covered yet
        if k in ignore_list or len(tree.leaves()) > max_sent_length: #31:
            ignored_sents.append(k)
            continue

        with_phrase_labels = parsedargs.with_phrase_labels
        preproc_sent, bies_labels = biesLabels(tree, tokenizer, with_phrase_labels=with_phrase_labels, skip_unkown_tokens=True)

        assert len(preproc_sent) == len(bies_labels)

        # Do not store the same sentence twice!
        if parsedargs.duplicates:
            if str(preproc_sent) in output_sents:
                print('Found duplicate sentence, skipping: ', preproc_sent)
                continue
        output_sents.add(str(preproc_sent))

        for l in bies_labels:
            if l not in label_counts:
                label_counts[l] = 0
            label_counts[l] = label_counts[l] + 1

        bies_labels_file.write(' '.join(bies_labels) + '\n')
        text_toks_file.write(' '.join(preproc_sent) + '\n')

    text_toks_file.close()
    bies_labels_file.close()

    print('Finished. ignored sentences: ', ignored_sents)
    print('Label distribution: ')
    label_counts = dict(sorted(label_counts.items(), key=lambda item: item[1], reverse=True))
    total = sum(label_counts.values())
    print(label_counts)
    for l,c in label_counts.items():
        print(l,'\t', c, '\t', str(float(c/total)))