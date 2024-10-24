cd ../src/berkeley_parser

mkdir -p ../../resources/grammars/berkeley/$1

head -$1 ../../resources/treebanks/original_treebank_500k.txt > ../../resources/treebanks/$1.txt 

java -Xmx4g -cp CustomBerkeley.jar edu.berkeley.nlp.PCFGLA.GrammarTrainer \
    -path ../../resources/treebanks/$1.txt \
    -out ../../resources/grammars/berkeley/$1/$1 \
    -treebank SINGLEFILE \
    -mergingPercentage $2 \
    -filter 1.0e-5 \
    -SMcycles $3  \
    -rare 0 \
    -reallyRare 0

java -Xmx4g -cp BerkeleyParser-1.5.jar edu.berkeley.nlp.PCFGLA.WriteGrammarToTextFile \
    ../../resources/grammars/berkeley/$1/$1 \
    ../../resources/grammars/berkeley/$1/$1
    
wc -l ../../resources/grammars/berkeley/$1/$1.grammar

python process_grammar.py \
    -p ../../resources/grammars/berkeley/$1/$1 \
    -o ../../resources/grammars/nltk/$1 \
    -t 1e-7
