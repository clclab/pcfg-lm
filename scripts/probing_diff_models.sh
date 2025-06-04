#!/bin/bash

# Exit immediately if any command exits with a non-zero status.
set -e

# Huggingface hub model id
HUB_MODEL_ID="jumelet/gpt2_10000t_1M_256d_8l"
# Size of the treebank used to train the LLM
TREEBANK_SIZE="10000"
# Directory where to store the trained probing models
PROBING_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/constprobing-paper-version/models
# Directory where the input data is stored
DATA_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/constprobing-paper-version/data
# Device to use for training, if not specified, the script will use the first available GPU
DEVICE=cpu
# Directory where the output of the probing experiments will be stored
OUTPUT_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/constprobing-paper-version/results

# loop over experiments
experiment_type=("lca_tree" "shared_levels" "unary" "chunking" "lca" )
for etype in "${experiment_type[@]}"; do
    echo "Running probing experiments for treebank size ${TREEBANK_SIZE} with model ${HUB_MODEL_ID}"
    python main.py    --trainer.device ${DEVICE} \
                        --data.rel_toks ${DATA_DIR}/${TREEBANK_SIZE}/train_rel_toks.txt \
                        --data.pos_tags ${DATA_DIR}/${TREEBANK_SIZE}/train_POS_labels.txt \
                        --data.gold_trees ${DATA_DIR}/${TREEBANK_SIZE}/gold_trees_cleaned.txt \
                        --data.sentences ${DATA_DIR}/${TREEBANK_SIZE}/train_text.txt \
                        --data.data_dir ${DATA_DIR}/${TREEBANK_SIZE} \
                        --data.output_dir ${OUTPUT_DIR}/${TREEBANK_SIZE} \
                        --experiments.checkpoint_path ${PROBING_DIR}/${TREEBANK_SIZE} \
                        --activations.output_dir ${DATA_DIR}/${TREEBANK_SIZE}/ \
                        --experiments.type ${etype} \
                        --results.confusion_matrix \
                        --trainer.epochs 20
done

