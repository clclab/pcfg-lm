for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

python src/data_generation/generate.py \
    -g resources/grammars/nltk/${TREEBANK_SIZE}_pcfg.txt \
    -o ${DATA_OUTPUT:-resources/corpora/${TREEBANK_SIZE}} \
    --min_length ${MIN_LENGTH:-3} \
    --max_length ${MAX_LENGTH:-30} \
    --corpus_size ${CORPUS_SIZE:-10000} \
    --split_ratio ${SPLIT_RATIO:-0.8/0.1/0.1} \
    --store_trees
