import nltk
from transformers import PreTrainedModel
from utils import load_model_tokenizer
from tqdm import *
from tokenizer import *
import argparse
import torch
from pathlib import Path
import random
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import pickle
import os
import logging

logger = logging.getLogger(__name__)

def tree_to_pos(tree, skip_unk_tokens=False):
    pos_tags = [
        prod.lhs().symbol().split("_")[0]
        for prod in tree.productions()
        if isinstance(prod.rhs()[0], str)
    ]
    assert len(pos_tags) == len(tree.leaves())
    if skip_unk_tokens:
        no_unk_pos = []
        for pos, w in zip(pos_tags, tree.leaves()):
            if w in tokenizer.vocab:
                no_unk_pos.append(pos)
        return no_unk_pos
    else:
        return pos_tags

def create_states(
    tokenizer, 
    tree_corpus, 
    model, 
    device,
    concat=True, 
    skip_cls=False, 
    num_items=None,
    verbose=False,
    all_layers=True,
    skip_unk_tokens=False,
):
    if isinstance(model, PreTrainedModel):
        all_sens = [torch.tensor(tokenizer.convert_tokens_to_ids(tree.leaves())) for tree in tree_corpus]
        pad_idx = tokenizer.pad_token_id
        num_parameters = model.num_parameters()
    else:
        all_sens = [tokenizer.tokenize(tree.leaves(), pos_tags=tree_to_pos(tree)) for tree in tree_corpus]
        pad_idx = tokenizer.pad_idx
        num_parameters = model.num_parameters

    if num_items is not None:
        all_sens = random.sample(all_sens, num_items)

    lengths = [len(sen) for sen in all_sens]
    sen_tensor = pad_sequence(all_sens, padding_value=pad_idx, batch_first=True).to(device)

    batch_size = int(1e9 / num_parameters)
    states = defaultdict(list) if all_layers else []
    iterator = range(0, len(all_sens), batch_size)
    if verbose:
        iterator = tqdm(iterator)

    for idx in iterator:
        batch = sen_tensor[idx: idx + batch_size]

        with torch.no_grad():
            all_hidden = model(batch, output_hidden_states=True).hidden_states

        if all_layers:
            for layer_idx, layer_hidden in enumerate(all_hidden):
                for hidden, sen, length in zip(layer_hidden, batch, lengths[idx: idx + batch_size]):
                    unk_mask = sen[:length] != tokenizer.unk_token_id
                    states[layer_idx].append(hidden[:length][unk_mask])
        else:
            states.extend([
                hidden[int(skip_cls):length]
                for hidden, length in zip(all_hidden[-1], lengths[idx: idx + batch_size])
            ])

    if concat:
        if all_layers:
            for layer_idx, layer_states in states.items():
                states[layer_idx] = torch.concat(layer_states)
            return states
        else:
            return torch.concat(states)
    else:
        return states

if __name__ == '__main__':
    """
    Run with: python create_activations.py --checkpoint deberta
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default='data')
    parser.add_argument('--output_dir', type=Path, default='data')
    parser.add_argument('--model_id', required=True)
    parser.add_argument('--concat', action='store_true')
    parsedargs = parser.parse_args()

    # Load data
    with open(parsedargs.data_dir) as f:
        tree_corpus = [nltk.Tree.fromstring(l.strip()) for l in f]

    parsedargs.output_dir.mkdir(parents=True, exist_ok=True)

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

    all_test_mccs = []

    # Load model
    model, tokenizer = load_model_tokenizer(parsedargs)
    model.to(device)
    model.eval()

    # extract hidden states from the model
    all_layer_states = create_states(
        tokenizer, 
        tree_corpus, 
        model, 
        device,
        concat=parsedargs.concat, 
        skip_cls=False, 
        verbose=True, 
        all_layers=True, 
        skip_unk_tokens=True
    )

    # store all_layer_states in pickle
    if parsedargs.concat:
        with open(parsedargs.output_dir / 'activations_concat_layers.pickle', 'wb') as f:
            pickle.dump(all_layer_states, f)
    else:
        with open(parsedargs.output_dir / 'activations.pickle', 'wb') as f:
            pickle.dump(all_layer_states, f)

