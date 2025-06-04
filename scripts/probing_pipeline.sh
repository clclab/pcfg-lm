# probing prepare data
# probing diff models
# probing predictions to trees
# probing eval trees

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

bash scripts/probing_prepare_data.sh

bash scripts/probing_diff_models.sh

bash scripts/probing_predictions_to_trees.sh

bash scripts/probing_eval_trees.sh