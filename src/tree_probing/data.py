import torch
import numpy as np
import pickle
from pathlib import Path
import logging
from collections import Counter
import json

logger = logging.getLogger(__name__)

class ExperimentManager():
    """
    This class is used to load the data and activations for the different experiments.
    """
    def __init__(self, config_dict):
        """
        If you add a new experiment:
        1. Add the label path to the _set_label_path method
        2. Add the experiment in the right category in _load_activations
        """
        self.config_dict = config_dict
        self.name = config_dict['experiments']['type']
        self.device = config_dict['trainer']['device']
        self.rel_toks = self._read_rel_toks()
        self.sentences = self._read_sentences()
        self.label_path = self._set_label_path()
        self.labels, self.label_vocab = self._create_labels()
        self.idx2class = self._create_idx2class()
        self._set_results_file()
        self.activations = self._load_activations()


    def _create_idx2class(self):
        return {v: k for k, v in self.label_vocab.items()}

    def _create_labels(self):
        logging.info(f'Creating labels for {self.name}')
        labels = []

        counter = 0
        with open(self.label_path, 'r') as f:
            for line in f:
                counter += 1
                labels.extend(line.strip().split())

        vocab = {l: idx for idx, l in enumerate(set(labels))}
        
        tokenized_labels = torch.tensor([vocab[l] for l in labels]).to(self.device)

        return tokenized_labels, vocab

    def create_train_dev_test_split(self, idx, train_size=0.8, dev_size=0.9):
        states = self.activations[idx]
        total_size = len(states)

        train_idx, dev_idx, test_idx = int(total_size * train_size), int(total_size * dev_size), total_size
        
        train_ids = range(0, train_idx)
        dev_ids = range(train_idx, dev_idx)
        test_ids = range(dev_idx, test_idx)

        X_train = states[train_ids]
        y_train = self.labels[train_ids]

        X_dev = states[dev_ids]
        y_dev = self.labels[dev_ids]

        X_test = states[test_ids]
        y_test = self.labels[test_ids]

        if self.name in ['lca_tree', 'shared_levels', 'unary']:
            self.rel_toks_test = [self.rel_toks[idx] for idx in test_ids]

        return X_train, y_train, X_dev, y_dev, X_test, y_test

    def _load_activations(self):
        """
        1. Activations per layer
        2. Activations concatenated into one
        3. Sampled activations
        """
        
        logging.info(f'Loading activations for {self.name}')
        # Set path to activations
        activations_path = self._set_activations_path()

        # Loading activations
        if activations_path != None and activations_path.exists():
            with open(activations_path, 'rb') as f:
                activations = pickle.load(f)
        else:
            logging.critical(f"Loading of activations failed (path: {activations_path}), check path or generate with create_activations.py or combine_activations.py")
            raise ValueError('Loading of activations failed')


        if self.name in ['lca', 'lca_tree', 'shared_levels', 'unary']:
            for layer_idx, layer_states in activations.items():
                activations[layer_idx] = torch.concat(layer_states)

        assert len(self.labels) == len(activations[0]), \
        f"Length of labels ({len(self.labels)}) does not match length of activations ({len(activations[0])})"

        return activations
    

    def _read_rel_toks(self):
        with open(self.config_dict['data']['rel_toks'], 'r') as f:
            rel_toks = f.readlines()
        
        return [tok.strip('\n') for sent in rel_toks for tok in sent.split(' ')]

    
    def _read_sentences(self):
        with open(self.config_dict['data']['sentences'], 'r') as f:
            sentences = f.readlines()
        
        return [sent.strip('\n') for sent in sentences]
    
    def _set_activations_path(self):
        # Set activation path based on experiment
        if self.name == 'chunking':
            activations_path = self.config_dict['activations']['output_dir'] / 'activations_concat_layers.pickle'
        
        elif self.name == 'lca':
            activations_path = self.config_dict['activations']['output_dir'] / 'activations_layers_combined.pickle'
            
        elif self.name in ['lca_tree', 'shared_levels', 'unary']:
            activations_path = self.config_dict['activations']['output_dir'] / 'activations_concat_combined.pickle'
            
        else:
            logging.critical(f"This experiment is not supported yet: {self.name}.")
            return None
        
        return activations_path

    def _set_label_path(self):
        if self.name == 'chunking':
            logging.info('Running chunking experiments')
            label_path = self.config_dict['data']['data_dir'] / 'train_bies_labels.txt'

        elif self.name == 'lca':
            logging.info('Running lca experiments')
            label_path = self.config_dict['data']['data_dir'] / 'train_rel_labels.txt'

        elif self.name == 'lca_tree':
            logging.info('Running lca for full reconstructing (this means, concatenated layers!)')
            label_path = self.config_dict['data']['data_dir'] / 'train_rel_labels.txt'

        elif self.name == 'shared_levels':
            logging.info('Running shared levels for full reconstructing WITHOUT sampling (this means, concatenated layers!)')
            label_path = self.config_dict['data']['data_dir'] / 'train_shared_levels.txt'

        elif self.name == 'unary':
            logging.info('Running unary experiments for full reconstruction.')
            label_path = self.config_dict['data']['data_dir'] / 'train_unaries.txt'

        else:
            logging.critical("This experiment is not supported yet.")
            raise ValueError('This experiment is not supported yet.')
        
        return label_path

    def _set_results_file(self):
        results_dir = self.config_dict['data']['output_dir'] / f'{self.name}'
        results_dir.mkdir(parents=True, exist_ok=True)

        self.results_file = open(results_dir / f'results_default.txt', 'w')
        self.test_results_file = self.config_dict['data']['output_dir'] /f'{self.name}/test_results_default.pickle'
        self.val_results_file = self.config_dict['data']['output_dir'] /f'{self.name}/val_results_default.pickle'
        self.base_name = f'{self.name}/best_model_{self.name}'
    



