#! /bin/bash
# Launch a batched job running a bash script within a container.
# Special arguments used:
# --time=24:00:00 : Run for max of 24 hours (hours, minutes, seconds). To run 2 days: 2-00:00:00
# -A es_hutter : Run on RSL's member share of Euler
# --mail-type=END : Triggers email to $USER@ethz.ch at the end of training
# --warp: run command in ''. Does not work with variables (that's why run.sh is needed).

export run_cmd="python3 /deXtreme/IsaacGymEnvs/isaacgymenvs/train.py \
                  task=FrankaCubeStack \
                  wandb_activate=True \
                  wandb_entity=joarder-arunim-github \
                  headless=True \
                  experiment=euler_franka_cube_stack_base"

# python3 train.py task=Ant wandb_activate=True wandb_entity=joarder-arunim-github headless=True experiment=euler_ant_base

export custom_flags="--nv --writable -B /cluster/home/$USER/git/IsaacGymEnvs/:/deXtreme/IsaacGymEnvs"

sbatch \
  -n 8 \
  --mem-per-cpu=8192 \
  -G 1 \
  --gres=gpumem:204800 \
  --time=15:00:00 \
  -A es_hutter \
  --mail-type=END \
  --tmp=15G \
  run.sh 
