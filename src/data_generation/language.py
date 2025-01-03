from collections import Counter
import pickle
import random
import os

from copy import deepcopy
from typing import Tuple, Optional, TypeVar, Generic, List, Dict

from nltk import Tree, PCFG as nltk_PCFG

from config import Config


class LanguageConfig(Config):
    split_ratio: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    corpus_size: Optional[int] = None
    allow_duplicates: bool = False
    file: Optional[str] = None
    store_trees: bool = True
    test_on_unique: bool = False


C = TypeVar("C", bound=LanguageConfig)


class Language(Generic[C]):
    def __init__(self, config: C):
        assert sum(config.split_ratio) == 1.0, "train/dev/test split does not add to 1"

        self.config = config
        self.grammar = self.create_grammar()

        if self.config.file is not None:
            with open(self.config.file, "rb") as f:
                self.corpus, self.tree_corpus, self.pos_dict = pickle.load(
                    self.config.file
                )

            if self.config.corpus_size and self.config.corpus_size < len(self.corpus):
                # This allows us to subsample the larger corpus directly for smaller corpus sizes
                self.corpus = self.corpus[: self.config.corpus_size]
                self.tree_corpus = {sen: self.tree_corpus[sen] for sen in self.corpus}
                self.pos_dict = {sen: self.pos_dict[sen] for sen in self.corpus}
        else:
            self.tree_corpus: Dict[str, Tree] = {}
            self.corpus = self.create_corpus()

        self.train_corpus, self.dev_corpus, self.test_corpus = self.split()

    def __len__(self):
        return len(self.train_corpus) + len(self.dev_corpus) + len(self.test_corpus)

    def save(self, file_name: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump((self.corpus, self.tree_corpus, self.pos_dict), file_name)

    def create_grammar(self) -> Optional[nltk_PCFG]:
        pass

    def create_corpus(self) -> List[str]:
        raise NotImplementedError

    def split(self) -> Tuple[List[str], List[str], List[str]]:
        random.shuffle(self.corpus)
        train_ratio, dev_ratio, test_ratio = self.config.split_ratio

        if self.config.test_on_unique:
            item_distribution = Counter(self.corpus)
            unique_items = list(item_distribution.keys())

            train_split_idx = int(len(item_distribution) * train_ratio)
            dev_split_idx = int(len(item_distribution) * (train_ratio + dev_ratio))
            test_split_idx = int(
                len(item_distribution) * (train_ratio + dev_ratio + test_ratio)
            )

            train_items = unique_items[:train_split_idx]
            dev_items = unique_items[train_split_idx:dev_split_idx]
            test_items = unique_items[dev_split_idx:test_split_idx]

            # Duplicate each unique item according to the original counts of the item
            train_items = [
                x for item in train_items for x in [item] * item_distribution[item]
            ]
            dev_items = [
                x for item in dev_items for x in [item] * item_distribution[item]
            ]
            test_items = [
                x for item in test_items for x in [item] * item_distribution[item]
            ]
        else:
            train_split_idx = int(len(self.corpus) * train_ratio)
            dev_split_idx = int(len(self.corpus) * (train_ratio + dev_ratio))
            test_split_idx = int(
                len(self.corpus) * (train_ratio + dev_ratio + test_ratio)
            )

            train_items = self.corpus[:train_split_idx]
            dev_items = self.corpus[train_split_idx:dev_split_idx]
            test_items = self.corpus[dev_split_idx:test_split_idx]

        return train_items, dev_items, test_items
        
    def store(self, output: str):
        if not os.path.exists(output):
            os.makedirs(output)

        if self.config.split_ratio[0] > 0.0:
            train_path = os.path.join(output, "train.txt")
            self.store_corpus(train_path, self.train_corpus)

        if self.config.split_ratio[1] > 0.0:
            dev_path = os.path.join(output, "dev.txt")
            self.store_corpus(dev_path, self.dev_corpus)

        if self.config.split_ratio[2] > 0.0:
            test_path = os.path.join(output, "test.txt")
            self.store_corpus(test_path, self.test_corpus)
                
    def store_corpus(self, path: str, corpus: List[str]):
        with open(path, "w") as f:
            f.write("\n".join(corpus))
        
        if self.config.store_trees:
            trees = [self.tree_corpus[sen] for sen in corpus]
            str_trees = [' '.join(str(tree).split()) for tree in trees]
            
            with open(path.replace("txt", "nltk"), "w") as f:
                f.write('\n'.join(str_trees))

