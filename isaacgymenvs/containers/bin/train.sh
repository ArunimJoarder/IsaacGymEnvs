#! /bin/bash
# Launch a batched job running a bash script within a container.
# Special arguments used:
# --time=24:00:00 : Run for max of 24 hours (hours, minutes, seconds). To run 2 days: 2-00:00:00
# -A es_hutter : Run on RSL's member share of Euler
# --mail-type=END : Triggers email to $USER@ethz.ch at the end of training
# --warp: run command in ''. Does not work with variables (that's why run.sh is needed).

export run_cmd="python3 /IsaacGymEnvs/isaacgymenvs/train.py \
                  task=AllegroHandDextremeAdversarialObservations \
                  base_checkpoint=exported_models/AllegroHandADR.onnx \
                  wandb_activate=True \
                  wandb_entity=joarder-arunim-github \
                  headless=True \
                  num_envs=8192 \
                  experiment=train_adv_obs_cube_pos_0.05_cube_rot_0.1_no_dof_pos_obs \
                  task.experiment_dir=while_training \
                  task.env.printNumSuccesses=True"

export custom_flags="--nv --writable -B /cluster/home/$USER/git/IsaacGymEnvs/:/deXtreme/IsaacGymEnvs"

sbatch \
  -n 16 \
  --mem-per-cpu=2048 \
  -G 1 \
  --gres=gpumem:10240 \
  --time=12:00:00 \
  -A es_hutter \
  --mail-type=END \
  --tmp=15G \
  run.sh 
