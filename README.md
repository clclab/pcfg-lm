# Transparency at the Source
Repo for the EMNLP 2023 Findings paper _Transparency at the Source: Evaluating and Interpreting Language Models With Access to the True Distribution_.

Our pipeline consists of a mix of Java and Python code: grammar induction is done using the original code of Petrov et al. (2006) in Java, language model training is done using `transformers` in Python.

**Grammar induction command** 
`java -Xmx32g -cp CustomBerkeley.jar edu.berkeley.nlp.PCFGLA.GrammarTrainer -path $path -out $save_dir/stage -treebank SINGLEFILE -mergingPercentage 0.5 -filter 1.0e-8 -SMcycles 5`

EarleyX causal PCFG probabilities command:
`java -Xms32768M -classpath "earleyx_fast.jar:lib/*" parser.Main -in data/eval_subset_100.txt -grammar grammars/1.0_petrov.grammar -out results -verbose 1 -thread 1`

Language model training command:
`python3 main_multi.py --model.model_type microsoft/deberta-base --model.is_mlm --tokenizer.path tokenizers/added_tokens.json --data.data_dir corpora --data.train_file train.txt --trainer.output_dir $save_dir`
