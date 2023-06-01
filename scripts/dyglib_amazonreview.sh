#!/bin/bash

#SBATCH --account=def-rrabba
#SBATCH --time=5-20:00:00               # time (DD-HH:MM)
#SBATCH --cpus-per-task=4               # CPU cores/threads
#SBATCH --gres=gpu:1                    # number of GPU(s) per node
#SBATCH --mem=100G                      # memory (per node)
#SBATCH --job-name=DyGLib_amazonreview
#SBATCH --output=outlog/%x-%j.log



# model_name="GraphMixer"
dataset_name="amazonreview"
seed=2023
num_runs=5


for model_name in TCL CAWN TGAT JODIE DyGFormer DyRep 
do 
    start_time="$(data -u +%u)"

    echo " >>> MODEL: $model_name"
    echo " >>> DATA: $dataset_name"
    echo " >>> Num. Runs: $num_runs"
    echo " >>> Seed: $seed"

    echo "===================================================================================="
    echo "===================================================================================="
    echo ""
    echo " ***** $model_name: $dataset_name *****"

    # command to run the model
    python train_link_pred_trans_tgb.py --dataset_name "$dataset_name" --model_name "$model_name" --seed "$seed" --num_runs "$num_runs"

    end_time="$(date -u +%s)"
    elapsed="$(($end_time-$start_time))"
    echo "Model: $model_name, Data: $data: Elapsed Time: $elapsed seconds."
    echo ""
    echo "===================================================================================="
    echo "===================================================================================="

done