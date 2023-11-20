#!/bin/bash
#SBATCH --job-name=earleyx
#SBATCH -t 1:00:00
#SBATCH --partition=thin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32768

module load 2022
module load Java/11.0.16

cp -r $HOME/earleyx $TMPDIR/earleyx
mkdir $TMPDIR/earleyx2/new_results

cd $TMPDIR/earleyx
mkdir new_results

date

datafile=eval_subset_5

srun --exclusive java -Xms32768M -classpath "earleyx_fast.jar:lib/*" parser.Main -in data/${datafile}.txt -grammar grammars/1.0_petrov.grammar -out new_results/$datafile -verbose 1 -thread 1

date

mkdir $HOME/earleyx/eval_results5
cp -r $TMPDIR/earleyx/new_results/* $HOME/earleyx/eval_results5/

