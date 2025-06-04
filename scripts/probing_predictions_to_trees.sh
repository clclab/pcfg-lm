#!/bin/bash

# Exit immediately if any command exits with a non-zero status.
set -e

# Size of the treebank used to train the LLM
TREEBANK_SIZE="10000"
DATA_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/constprobing-paper-version/results

echo "Creating trees for treebank size=$TREEBANK_SIZE"
current_output_dir=$DATA_DIR/$TREEBANK_SIZE/full_tree
mkdir -p $current_output_dir
python predictions_to_trees.py --lca $DATA_DIR/$TREEBANK_SIZE/lca_tree/predictions_lca_tree.txt \
                                --levels $DATA_DIR/$TREEBANK_SIZE/shared_levels/predictions_shared_levels.txt \
                                --out $current_output_dir/concat_test_trees.txt \
                                --pos_text $DATA_DIR/$TREEBANK_SIZE/lca_tree/test_POS_tags.txt \
                                --unary $DATA_DIR/$TREEBANK_SIZE/unary/predictions_unary.txt


