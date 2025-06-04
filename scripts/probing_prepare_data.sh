#!/bin/bash

# Exit immediately if any command exits with a non-zero status.
set -e

####################################################################################################
## LABELS
####################################################################################################

DATA_DIR="../corpora"
TREEBANK_SIZE="50"
HUB_MODEL_ID="jumelet/gpt2_50t_1M_256d_8l"
OUTPUT_DIR="../data"
mkdir -p ${OUTPUT_DIR}/${TREEBANK_SIZE}

# Remove labels from the gold trees
echo "Starting cleaning gold trees from integer specifications..."
python clean_gold_trees.py --data_dir ${DATA_DIR}/${TREEBANK_SIZE}/test.nltk \
    --output_dir ${OUTPUT_DIR}/${TREEBANK_SIZE}/gold_trees_cleaned.txt
echo "Done cleaning gold trees from integer specifications."

# extract bies labels (chunking) from the gold trees
echo "Starting extracting bies labels from the gold trees..."
python extract_bies_labels.py --model_id ${HUB_MODEL_ID} \
    --data_dir ${DATA_DIR}/${TREEBANK_SIZE}/test.nltk \
    --text_toks ${OUTPUT_DIR}/${TREEBANK_SIZE}/train_text_bies.txt \
    --bies_labels ${OUTPUT_DIR}/${TREEBANK_SIZE}/train_bies_labels.txt
echo "Done extracting bies labels from the gold trees."

# extract lca + shared level labels from the gold trees
echo "Starting extracting lca + shared level labels from the gold trees..."
python extract_lca_labels.py --model_id ${HUB_MODEL_ID} \
    --data_dir ${DATA_DIR}/${TREEBANK_SIZE}/test.nltk \
    --text_toks ${OUTPUT_DIR}/${TREEBANK_SIZE}/train_text.txt \
    --rel_toks ${OUTPUT_DIR}/${TREEBANK_SIZE}/train_rel_toks.txt \
    --rel_labels ${OUTPUT_DIR}/${TREEBANK_SIZE}/train_rel_labels.txt \
    --next \
    --shared_levels ${OUTPUT_DIR}/${TREEBANK_SIZE}/train_shared_levels.txt \
    --unary ${OUTPUT_DIR}/${TREEBANK_SIZE}/train_unaries.txt
echo "Done extracting lca + shared level labels from the gold trees."

# extract POS labels
echo "Starting extracting POS labels from the gold trees..."
python extract_POS.py --model_id ${HUB_MODEL_ID} \
            --data_dir ${DATA_DIR}/${TREEBANK_SIZE}/test.nltk \
            --output_dir ${OUTPUT_DIR}/${TREEBANK_SIZE}/train_POS_labels.txt

# ####################################################################################################
# ## EXTRACT ACTIVATIONS
# ####################################################################################################

# extract wordlevel activations from the model PER LAYER
python create_activations.py --model_id ${HUB_MODEL_ID} \
    --data_dir ${DATA_DIR}/${TREEBANK_SIZE}/test.nltk \
    --output_dir ${OUTPUT_DIR}/${TREEBANK_SIZE}

# extract wordlevel activations from the model CONCATENATED
python create_activations.py --model_id ${HUB_MODEL_ID} \
    --data_dir ${DATA_DIR}/${TREEBANK_SIZE}/test.nltk \
    --output_dir ${OUTPUT_DIR}/${TREEBANK_SIZE} \
    --concat

# combine wordlevel activations to pairlevel activations PER LAYER
python combine_activations.py --input_file ${OUTPUT_DIR}/${TREEBANK_SIZE}/activations.pickle \
    --output_file ${OUTPUT_DIR}/${TREEBANK_SIZE}/activations_layers_combined.pickle \
    --rel_toks ${OUTPUT_DIR}/${TREEBANK_SIZE}/train_rel_toks.txt \
    --mode concat

# combine wordlevel activations to pairlevel activations WITH CONCATENATED LAYERS
python combine_activations.py --input_file ${OUTPUT_DIR}/${TREEBANK_SIZE}/activations_layers_combined.pickle \
    --output_file ${OUTPUT_DIR}/${TREEBANK_SIZE}/activations_concat_combined.pickle \
    --rel_toks ${OUTPUT_DIR}/${TREEBANK_SIZE}/train_rel_toks.txt \
    --mode concat \
    --concatenate_layers

