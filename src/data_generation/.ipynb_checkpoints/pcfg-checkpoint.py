import random
from typing import List, Optional

from nltk import Tree, PCFG as nltk_PCFG, Nonterminal
from nltk.parse import ChartParser, IncrementalLeftCornerChartParser
from tqdm import tqdm

from language import Language, LanguageConfig


class PCFGConfig(LanguageConfig):
    grammar_file: str
    max_length: int
    max_depth: int
    min_length: int = 0
    start: Optional[str] = None
    generation_factor: int = 10
    verbose: bool = True


class PCFG(Language[PCFGConfig]):
    def __repr__(self):
        return str(self.grammar)

    def create_corpus(self) -> List[str]:
        return self._generate_corpus(self.grammar)

    def create_grammar(self) -> nltk_PCFG:
        print("Loading grammar...")
        with open(self.config.grammar_file) as f:
            raw_grammar = f.read()
        grammar = nltk_PCFG.fromstring(raw_grammar)

        if self.config.start is not None:
            grammar._start = Nonterminal(self.config.start)

        grammar._lhs_prob_index = {}
        for lhs in grammar._lhs_index.keys():
            lhs_probs = [prod.prob() for prod in grammar.productions(lhs=lhs)]
            grammar._lhs_prob_index[lhs] = lhs_probs
        
        print("Grammar loaded!")
        
        return grammar

    def _generate_corpus(self, grammar: nltk_PCFG) -> List[str]:
        """
        We divide the generation in an inner and outer loop:
        The outer loop sets up a new generation procedure, the inner loop
        determines how many items we sample from a top-down approach,
        This outer/inner division appears to yield the least redundant generation.
        """
        str_corpus = []
        unique_items = set()

        total = self.config.corpus_size * self.config.generation_factor

        try:
            if self.config.verbose:
                print("Generating corpus...")
    
            with tqdm(total=self.config.corpus_size) as pbar:
                for _ in range(total):
                    tree = generate_tree(grammar, depth=self.config.max_depth)
                    item = tree.leaves()
                    item_len = len(item)

                    if self.config.min_length < item_len < self.config.max_length:
                        str_item = " ".join(item)
                        if not self.config.allow_duplicates and str_item in unique_items:
                            continue

                        pbar.update(1)
                        str_corpus.append(str_item)
                        unique_items.add(str_item)
                        
                        if self.config.store_trees:
                            self.tree_corpus[str_item] = tree
    
                    if len(str_corpus) >= self.config.corpus_size:
                        return str_corpus
        except KeyboardInterrupt:
            pass

        return list(str_corpus)

    def gen_parse(self, sen: List[str]):
        srp = ChartParser(self.grammar)

        for parse in srp.parse(sen):
            print(parse)

        return next(srp.parse(sen))


def generate_tree(grammar, start=None, depth=None, max_tries=10) -> Tree:
    if not start:
        start = grammar.start()
    if depth is None:
        depth = 100

    for _ in range(max_tries):
        try:
            tree_str = concatenate_subtrees(grammar, [start], depth)
            return Tree.fromstring(tree_str)
        except RecursionError:
            pass

    raise ValueError("No tree could be generated with current depth")


def concatenate_subtrees(grammar, items, depth):
    if items:
        children = []
        for item in items:
            children.append(generate_subtree(grammar, item, depth))

        return " ".join(children)
    else:
        return []


def generate_subtree(grammar, lhs, depth):
    if depth > 0:
        if isinstance(lhs, Nonterminal):
            productions = grammar.productions(lhs=lhs)
            probs = grammar._lhs_prob_index[lhs]

            for prod in random.choices(productions, probs, k=1):
                children = concatenate_subtrees(grammar, prod.rhs(), depth - 1)
                return f"({lhs.symbol()} {children})"
        else:
            return lhs
    else:
        raise RecursionError


def cfg_str(prod):
    return Production.__str__(prod)


def rev_prod(prod):
    return Production(prod.lhs(), prod.rhs()[::-1])
