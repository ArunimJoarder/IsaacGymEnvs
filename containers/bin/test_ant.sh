#! /bin/bash
# Launch a batched job running a bash script within a container.
# Special arguments used:
# --time=24:00:00 : Run for max of 24 hours (hours, minutes, seconds). To run 2 days: 2-00:00:00
# -A es_hutter : Run on RSL's member share of Euler
# --mail-type=END : Triggers email to $USER@ethz.ch at the end of training
# --warp: run command in ''. Does not work with variables (that's why run.sh is needed).

export run_cmd="python3 /deXtreme/IsaacGymEnvs/isaacgymenvs/train.py \
                  task=AnymalFinetuningResidualActions \
                  test=True \
                  num_envs=1024 \
                  base_checkpoint=exported_models/Anymal.onnx \
                  checkpoint=runs/euler_anymal_flat_delta_ft_same_delta_net_size_2000_iters_27-10-13-48/nn/last_euler_anymal_flat_delta_ft_same_delta_net_size_2000_iters_ep_2000_rew__67.67_.pth \
                  headless=True \
                  experiment=last_euler_anymal_flat_delta_ft_same_delta_net_size_2000_iters_ep_2000_rew__67.67_-2 \
                  task.experiment_dir=delta_ft/anymal_flat \
                  task.env.printEvalStats=True"

#  \
                  # task.env.terrain.curriculum=False
# python3 train.py task=Ant wandb_activate=True wandb_entity=joarder-arunim-github headless=True experiment=euler_ant_base
# checkpoint=runs/euler_anymal_terrain_4500_iters_15-13-54-25/nn/last_euler_anymal_terrain_4500_iters_ep_4500_rew__19.31_.pth \

export custom_flags="--nv --writable -B /cluster/home/$USER/git/IsaacGymEnvs/:/deXtreme/IsaacGymEnvs,/cluster/home/$USER/git/rl_games/:/deXtreme/rl_games"

sbatch \
  -n 8 \
  --mem-per-cpu=8192 \
  -G 1 \
  --gres=gpumem:204800 \
  --time=01:00:00 \
  -A es_hutter \
  --mail-type=END \
  --tmp=15G \
  run.sh 
