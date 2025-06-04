import numpy as np
from typing import List, Dict

from transformers import AutoTokenizer, AutoModel
import os
import logging

logger = logging.getLogger(__name__)

def format_predictions(predictions: np.array, vocab: Dict, rel_toks: List) -> List[str]:
    """
    Format the predictions of a model to a list of strings.

    :param predictions: The predictions of the model as a 1D numpy array.
    :param vocab: A dictionary mapping the indices to the labels.
    :param sentence_lengths: A list of integers representing the sentence lengths.
    :return: A list of strings representing the predictions.
    """
    formatted_output = []
    sent = ''
    prev_line_sent_ix = rel_toks[0].split('_')[0]

    for i, (label, tok) in enumerate(zip(predictions.tolist(), rel_toks)):
        current_idx = tok.split('_')[0]

        if current_idx != prev_line_sent_ix:
            formatted_output.append(sent + '\n')
            sent = f'{vocab[label]} '
            prev_line_sent_ix = current_idx
        else:
            sent += f'{vocab[label]} '
    formatted_output.append(sent)

    return ''.join(formatted_output)


def idx_labels2text(result, label_vocab):
    f_result = {}
    idx2c = {v: k for k, v in label_vocab.items()}

    # output of pytorch lightning .test is a list with all logged metrics, in this case only one dict
    for c, acc in result[0].items():
        if c == 'test_acc' or c == 'val_acc':
            f_result[c] = acc
        else:
            class_label = int(c.split('_')[1])
            f_result[idx2c[class_label]] = acc

    return f_result


def load_tokenizer(args):
    return AutoTokenizer.from_pretrained(args.model_id)


def load_model_tokenizer(args):
    tokenizer = load_tokenizer(args)
    model = AutoModel.from_pretrained(args.model_id)

    return model, tokenizer


def word2sentenceformat(postextfile):
    with open(postextfile,'r') as f:
        text_and_pos = f.read().splitlines()
        wordsandpos = []
        w_p_sent = []
        for line in text_and_pos:
            if len(line) == 0:
                wordsandpos.append(w_p_sent)
                w_p_sent = []
                continue
            [w,p] = line.split()
            w_p_sent.append((w,p))
    
    return wordsandpos

def format_pos_and_write(pos_corpus, output_file):
    """
    Format pos_corpus and write to output_file
    Input:
        pos_corpus: list of lists of POS tags
        output_file: str
    """
    with open(output_file, 'w') as f:
        for i, sentence in enumerate(pos_corpus):
            for j, (word, pos) in enumerate(sentence):
                f.write(f'{word} {pos}\n')
            f.write('\n')

def get_pos_test_set(config_dict, CurrentExperiment):
    pos_tags = word2sentenceformat(config_dict['data']['pos_tags'])

    # select the corresponding POS tags
    selected_pos_tags = []
    first_tok = CurrentExperiment.rel_toks_test[0].split('_')[0]
    partial_sentence = set()
    seen = set()
    for i, tok in enumerate(CurrentExperiment.rel_toks_test):
        sentence, word1, word2 = tok.split('_')

        if sentence == first_tok:
            partial_sentence.update([word1, word2])
        else:
            if sentence not in seen:
                selected_pos_tags.append(pos_tags[int(sentence)])
                seen.add(sentence)

    partial_pos_tags = [pos_tags[int(first_tok)][int(index)] for index in partial_sentence]
    selected_pos_tags.insert(0, partial_pos_tags)
    format_pos_and_write(selected_pos_tags, f'{config_dict["data"]["output_dir"]}/{CurrentExperiment.name}/test_POS_tags.txt')

def get_gold_trees_test_set(config_dict, CurrentExperiment):
    # open txt file with gold trees
    with open(config_dict['data']['gold_trees'], 'r') as f:
        gold_trees = f.read().splitlines()
    
    # select the corresponding gold trees
    selected_gold_trees = [gold_trees[tok] for tok in set([int(tok.split('_')[0]) for tok in CurrentExperiment.rel_toks_test])]
    # write to file
    with open(f'{config_dict["data"]["output_dir"]}/{CurrentExperiment.name}/test_gold_trees.txt', 'w') as f:
        for tree in selected_gold_trees:
            f.write(tree + '\n')