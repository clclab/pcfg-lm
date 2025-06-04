import os
import nltk.tree
import argparse
from pathlib import Path
from utils import load_tokenizer
import torch
import json

def extract_POS(tree_corpus):
    """
    Extract POS tags from tree_corpus
    Input:
        tree_corpus: list of nltk.Tree
    Output:
        pos_corpus: list of lists of POS tags
    """
    pos_corpus = []
    for tree in tree_corpus:
        pos_corpus.append(tree.pos())

    return pos_corpus

def format_and_write(pos_corpus, tokenizer, output_file):
    """
    Format pos_corpus and write to output_file
    Input:
        pos_corpus: list of lists of POS tags
        output_file: str
    """
    with open(output_file, 'w') as f:
        for i, sentence in enumerate(pos_corpus):
            for j, (word, pos) in enumerate(sentence):
                if word not in tokenizer.vocab:
                    print('Skipping word not in tokenizer vocab: ', word)
                    continue

                f.write(f'{word} {pos.split("_")[0]}\n')
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='babyberta')
    parser.add_argument('--data_dir', type=Path, default=Path('corpora/eval_trees_10k.txt'))
    parser.add_argument('--output_dir', type=Path, default=Path('data/train_POS_v1.txt'))
    parser.add_argument('--specific_start', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    with open(args.data_dir, 'r') as f:
        tree_corpus = [nltk.tree.Tree.fromstring(l.strip()) for l in f]
            
    device = torch.device("cpu")
    tokenizer = load_tokenizer(args)
    
    POS_tags = extract_POS(tree_corpus)
    format_and_write(POS_tags, tokenizer, args.output_dir)

