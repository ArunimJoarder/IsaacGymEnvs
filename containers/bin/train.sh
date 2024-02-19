#! /bin/bash
# Launch a batched job running a bash script within a container.
# Special arguments used:
# --time=24:00:00 : Run for max of 24 hours (hours, minutes, seconds). To run 2 days: 2-00:00:00
# -A es_hutter : Run on RSL's member share of Euler
# --mail-type=END : Triggers email to $USER@ethz.ch at the end of training
# --warp: run command in ''. Does not work with variables (that's why run.sh is needed).

export run_cmd="python3 /deXtreme/IsaacGymEnvs/isaacgymenvs/train.py \
                  task=AllegroHandDextremeADRFinetuningResidualActions \
                  task.onnx_noise_gen_checkpoint=exported_models/AllegroHandAdversarialObservationsAndActions.onnx \
                  base_checkpoint=exported_models/AllegroHandADR.onnx \
                  wandb_activate=True \
                  wandb_entity=joarder-arunim-github \
                  headless=True \
                  num_envs=4096 \
                  experiment=euler_ft_delta_adv_prob_0.10_4096 \
                  task.experiment_dir=while_training \
                  task.env.printNumSuccesses=True \
                  task.env.adv_noise_prob=0.10"

export custom_flags="--nv --writable -B /cluster/home/$USER/git/IsaacGymEnvs/:/deXtreme/IsaacGymEnvs"

sbatch \
  -n 4 \
  --mem-per-cpu=8192 \
  -G 1 \
  --gres=gpumem:204800 \
  --time=4:00:00 \
  -A es_hutter \
  --mail-type=END \
  --tmp=15G \
  run.sh 
