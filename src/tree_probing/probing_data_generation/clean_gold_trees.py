import argparse
from pathlib import Path
import nltk
import re
import logging
import sys

logging.basicConfig(stream=sys.stdout,
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


def clean_labels(input_string):
    # Define a regular expression pattern to match the labels
    pattern = r'([A-Z]+)_\d+'
    
    # Use re.sub to replace the matched pattern with only the letters
    cleaned_string = re.sub(pattern, r'\1', input_string)

    return cleaned_string


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default=Path('corpora/eval_trees_10k.txt'))
    parser.add_argument('--output_dir', type=Path, default=Path('../data'))
    args = parser.parse_args()

    try:
        with open(args.data_dir, 'r') as f:
            tree_corpus = [l.strip() for l in f]
    except FileNotFoundError:
        logger.error(f"File not found: {args.data_dir}")
        sys.exit(1)
    
    cleaned_tree_corpus = [clean_labels(tree) for tree in tree_corpus]

    with open(args.output_dir, 'w') as f:
        # f.write('\n'.join(tree._pformat_flat("", "()", False) for tree in list(cleaned_tree_corpus)))
        f.write('\n'.join([tree for tree in cleaned_tree_corpus]))
    
