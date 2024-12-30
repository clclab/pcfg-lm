python src/lm_eval/process_earleyx_grammar.py -g $1 -o $2

# clear score files
rm -f $4.surprisal
rm -f $4.stringprob

java \
    -Xmx4g \
    -classpath "src/lm_eval/earleyx/earleyx_fast.jar:src/lm_eval/earleyx/lib/*" parser.Main \
    -in $3 \
    -grammar $2 \
    -out $4 \
    -verbose 0 \
    -thread 1

python src/lm_eval/eval_clm.py --model $5 --corpus $3 --pcfg_scores $4

