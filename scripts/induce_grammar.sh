mkdir -p resources/grammars/berkeley/$1

head -$1 resources/treebanks/original_treebank_500k.txt > resources/treebanks/$1.txt 

java -Xmx16g -cp src/data_generation/berkeley_parser/CustomBerkeley.jar edu.berkeley.nlp.PCFGLA.GrammarTrainer \
    -path resources/treebanks/$1.txt \
    -out resources/grammars/berkeley/$1/$1 \
    -treebank SINGLEFILE \
    -mergingPercentage $2 \
    -filter 1.0e-7 \
    -SMcycles $3  \
    -rare 0 \
    -reallyRare 0 > logs/$1_$2_$3.txt

java -Xmx16g -cp src/data_generation/berkeley_parser/BerkeleyParser-1.5.jar edu.berkeley.nlp.PCFGLA.WriteGrammarToTextFile \
    resources/grammars/berkeley/$1/$1 \
    resources/grammars/berkeley/$1/$1
    
wc -l resources/grammars/berkeley/$1/$1.grammar

python src/data_generation/process_berkeley_grammar.py \
    -p resources/grammars/berkeley/$1/$1 \
    -o resources/grammars/nltk/$1 \
    -t 1e-7
