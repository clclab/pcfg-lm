# Scripts for running the whole pipeline

## Pipeline
The full pipeline, from grammar induction to data generation to model training to evaluation, can be run as follows:
```
pipeline.sh TREEBANK_SIZE=10 MERGE_PERCENTAGE=0.9 SM_CYCLES=3
```

We use a named argument setup of the form `$NAME=$VALUE`, see the `pipeline.sh` script for which arguments can be passed (related to all configuration of the grammar and LM architecture).

`$TREEBANK_SIZE` sets the number of parse trees from the treebank we induce a grammar from, by taking the first `n` trees of the treebank.

## Individual Modules
Script for grammar induction based on a treebank:
```
induce_grammar.sh $TREEBANK_SIZE $MERGE_PERCENTAGE $SM_CYCLES
```

Script for generating a corpus from the induced grammar (this example is set for a grammar induced from the first 2 treebank items):
```
python src/data_generation/generate.py \
    -g resources/grammars/nltk/2_pcfg.txt \
    -o resources/corpora/2 \
    --min_length 3 \
    --max_length 28 \
    --corpus_size 10000 \
    --split_ratio 0.8/0.1/0.1
```

Example script for training a MLM:
```
bash train_mlm.sh \
    --data.data_dir resources/corpora/2 \
    --trainer.output_dir resources/models/ \
    --model.hidden_size 16 \
    --model.num_hidden_layers 2 \
    --model.intermediate_size 16 \
    --model.num_attention_heads 2 \
    --trainer.num_train_epochs 2
```

Example script for evaluating a trained (causal) LM:
```
bash scripts/clm_eval.sh \
    resources/grammars/nltk/2_pcfg.txt \
    resources/grammars/earleyx/2.grammar \
    resources/corpora/2/test.txt \
    resources/evaluation/2 \
    resources/models/2/checkpoint-*/
```
