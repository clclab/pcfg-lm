from typing import List
import random

from nltk import Tree, CFG, Nonterminal

from language import Language, LanguageConfig


def tree_to_pos(tree, merge=True):
    if merge:
        return [
            prod.lhs().symbol().split("_")[0]
            for prod in tree.productions()
            if isinstance(prod.rhs()[0], str)
        ]
    else:
        return [
            prod.lhs().symbol()
            for prod in tree.productions()
            if isinstance(prod.rhs()[0], str)
        ]


class TreebankConfig(LanguageConfig):
    file: str
    max_length: int
    max_depth: int
    start: str
    min_length: int = 0
    sample: bool = False
    skip_root: bool = True


class Treebank(Language[TreebankConfig]):
    def __repr__(self):
        return str(self.config.file)

    def create_corpus(self) -> List[str]:
        with open(self.config.file) as f:
            lines = f.readlines()

        if self.config.sample:
            random.shuffle(lines)

        str_corpus = []

        for line in lines:
            tree = Tree.fromstring(line)

            if self.config.skip_root:
                tree = tree[0]

            if (
                tree.height() < self.config.max_depth
                and self.config.min_length
                <= len(tree.leaves())
                <= self.config.max_length
                and tree.label() == self.config.start
            ):
                str_item = self.tokenizer.config.sep_token.join(tree.leaves())
                str_corpus.append(str_item)
                self.tree_corpus[str_item] = tree
                if self.config.use_unk_pos_tags:
                    self.pos_dict[str_item] = tree_to_pos(tree)

            if (
                self.config.corpus_size is not None
                and len(str_corpus) >= self.config.corpus_size
            ):
                break

        unique_prods = set(
            prod for tree in self.tree_corpus.values() for prod in tree.productions()
        )
        self.grammar = CFG(Nonterminal(self.config.start), list(unique_prods))

        return str_corpus
