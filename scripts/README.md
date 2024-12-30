# Scripts for running the whole pipeline
Script for grammar induction based on a treebank:
```
induce_grammar.sh $TREEBANK_SIZE $MERGE_PERCENTAGE $SPLIT_MERGE_CYCLES
```

Script for generating a corpus from the induced grammar:
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
    resources/models/checkpoint-38/
```
