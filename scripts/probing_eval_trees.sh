#!/bin/bash

# Exit immediately if any command exits with a non-zero status.
set -e

# Set the path to the edge probing repository.
TREEBANK_SIZE="10000"
DATA_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/constprobing-paper-version/results
GOLD_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/constprobing-paper-version/data
EVALB_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/constprobing-paper-version

echo "Creating trees for treebank size=$TREEBANK_SIZE"
current_output_dir=$DATA_DIR/$TREEBANK_SIZE
current_data_dir=$GOLD_DIR/$TREEBANK_SIZE

echo "STEP: evaluate trees"

labeledoutput=$current_output_dir/full_tree/concat_evalb_labeled.log
unlabeledoutput=$current_output_dir/full_tree/concat_evalb_unlabeled.log

$EVALB_DIR/EVALB/evalb -p $EVALB_DIR/EVALB/COLLINS.prm $current_output_dir/full_tree/concat_test_trees.txt $current_output_dir/lca_tree/test_gold_trees.txt > $labeledoutput
$EVALB_DIR/EVALB/evalb -p $EVALB_DIR/EVALB/COLLINS_unlabeled.prm $current_output_dir/full_tree/concat_test_trees.txt $current_output_dir/lca_tree/test_gold_trees.txt > $unlabeledoutput

echo "STEP: Done with eval. Results are in the files: "
echo $labeledoutput
echo $unlabeledoutput
