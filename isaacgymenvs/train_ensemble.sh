#!/bin/bash

# exit if something fails!
set -e

experiment="default_ensemble"
num_models=5

while getopts e:n:m: flag
do
    case "${flag}" in
        e) experiment=${OPTARG};;
        m) num_models=${OPTARG};;
    esac
done

num_envs=4096
max_iterations=2000

for(( i=0; i<$num_models; i++ ))
do
    python train.py \
        task=AllegroHandDextremeAdversarialObservationsAndActionsEnsemble \
        base_checkpoint=exported_models/AllegroHandADR.onnx \
        headless=True \
        wandb_activate=True \
        wandb_entity=caviar-garnish \
        num_envs=$num_envs \
        max_iterations=$max_iterations \
        task.env.printNumSuccesses=False \
        task.experiment_dir="ensemble/${experiment}/model_${i}" \
        experiment="${experiment}" \
        +full_experiment_name="${experiment}_model_${i}" \
        task.model_num=${i} \
        task.ensemble_dir=ensemble_models \
        enable_ensemble=True

    python3 scripts/collect_checkpoints.py \
        -e ${experiment} \
        -m ${i} \
        -d runs/${experiment}_ensemble

    python3 train.py \
        task=AllegroHandDextremeAdversarialObservationsAndActionsEnsemble \
        base_checkpoint=exported_models/AllegroHandADR.onnx \
        checkpoint="runs/${experiment}_ensemble/${experiment}_${i}.pth" \
        headless=True \
        test=True \
        num_envs=2 \
        train.params.config.player.games_num=1 \
        experiment="${experiment}" \
        save_onnx=True \
        task.model_num=${i} \
        task.ensemble_dir=ensemble_models \
        enable_ensemble=True
done
