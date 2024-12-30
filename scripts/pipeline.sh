for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

bash scripts/induce_grammar.sh ${TREEBANK_SIZE} ${MERGE_PERCENTAGE} ${SM_CYCLES}

python src/data_generation/generate.py \
    -g resources/grammars/nltk/${TREEBANK_SIZE}_pcfg.txt \
    -o resources/corpora/${TREEBANK_SIZE} \
    --min_length ${MIN_LENGTH:-3} \
    --max_length ${MAX_LENGTH:-30} \
    --corpus_size ${CORPUS_SIZE:-10000} \
    --split_ratio 0.8/0.1/0.1
    
bash scripts/train_clm.sh \
    --data.data_dir resources/corpora/$TREEBANK_SIZE \
    --trainer.output_dir resources/models/$TREEBANK_SIZE \
    --model.hidden_size ${HIDDEN_SIZE:-16} \
    --model.num_hidden_layers ${NUM_HIDDEN_LAYERS:-2} \
    --model.intermediate_size ${INTERMEDIATE_SIZE:-16} \
    --model.num_attention_heads ${NUM_ATTENTION_HEADS:-2} \
    --trainer.num_train_epochs ${NUM_TRAIN_EPOCHS:-1} \
    --trainer.hub_model_id test \
    --trainer.hub_token hf_token.txt

bash scripts/clm_eval.sh \
    resources/grammars/nltk/${TREEBANK_SIZE}_pcfg.txt \
    resources/grammars/earleyx/${TREEBANK_SIZE}.grammar \
    resources/corpora/${TREEBANK_SIZE}/test.txt \
    resources/evaluation/${TREEBANK_SIZE} \
    resources/models/$TREEBANK_SIZE/checkpoint-*/
