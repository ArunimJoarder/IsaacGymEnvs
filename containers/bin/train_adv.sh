#! /bin/bash
# Launch a batched job running a bash script within a container.
# Special arguments used:
# --time=24:00:00 : Run for max of 24 hours (hours, minutes, seconds). To run 2 days: 2-00:00:00
# -A es_hutter : Run on RSL's member share of Euler
# --mail-type=END : Triggers email to $USER@ethz.ch at the end of training
# --warp: run command in ''. Does not work with variables (that's why run.sh is needed).

export run_cmd="python3 /deXtreme/IsaacGymEnvs/isaacgymenvs/train.py \
                  task=AllegroHandDextremeAdversarialObservationsAndActions \
                  base_checkpoint=exported_models/AllegroHandADR.onnx \
                  wandb_activate=True \
                  wandb_entity=caviar-garnish \
                  headless=True \
                  num_envs=8192 \
                  experiment=euler_adv_obs_acts \
                  task.experiment_dir=while_training \
                  task.env.printNumSuccesses=True \
                  max_iterations=4000"

# export run_cmd="python3 /deXtreme/IsaacGymEnvs/isaacgymenvs/train.py \
#                   task=AllegroHandDextremeADRFinetuningResidualActions \
#                   task.onnx_noise_gen_checkpoint=exported_models/AllegroHandAdversarialObservationsAndActions.onnx \
#                   base_checkpoint=exported_models/AllegroHandADR.onnx \
#                   headless=True \
#                   num_envs=1024 \
#                   experiment=euler_ft_delta_adv_prob_0.00_-1.0_action_penalty_-0.2_delta_action_penalty_adr_ranges_1024_5000_iters \
#                   task.experiment_dir=testt \
#                   task.env.printNumSuccesses=True \
#                   task.env.adv_noise_prob=0.00 \
#                   task.env.actionPenaltyScale=-1.0 \
#                   task.env.actionDeltaPenaltyScale=-0.2 \
#                   max_iterations=5000"

# python3 /deXtreme/IsaacGymEnvs/isaacgymenvs/train.py  task=AllegroHandDextremeADRFinetuningResidualActions  task.onnx_noise_gen_checkpoint=exported_models/AllegroHandAdversarialObservationsAndActions.onnx  base_checkpoint=exported_models/AllegroHandADR.onnx  headless=True  num_envs=1024  experiment=euler_ft_delta_adv_prob_0.00_-1.0_action_penalty_-0.2_delta_action_penalty_adr_ranges_1024_5000_iters  task.experiment_dir=testt  task.env.printNumSuccesses=True  task.env.adv_noise_prob=0.00  task.env.actionPenaltyScale=-10.0  task.env.actionDeltaPenaltyScale=-0.2  max_iterations=2000

# python3 /deXtreme/IsaacGymEnvs/isaacgymenvs/train.py task=AllegroHandDextremeADRFinetuningResidualActions task.onnx_noise_gen_checkpoint=exported_models/AllegroHandAdversarialObservationsAndActions.onnx base_checkpoint=exported_models/AllegroHandADR.onnx wandb_activate=True wandb_entity=joarder-arunim-github headless=True num_envs=1024 experiment=euler_ft_delta_adv_prob_0.00_-10.0_action_pen_-0.2_delta_action_pen_adr_ranges_1024_2000_iters_8_cpu task.experiment_dir=while_training task.env.printNumSuccesses=False task.env.adv_noise_prob=0.00 task.env.actionPenaltyScale=-10.0 task.env.actionDeltaPenaltyScale=-0.2 max_iterations=2000

export custom_flags="--nv --writable -B /cluster/home/$USER/git/IsaacGymEnvs/:/deXtreme/IsaacGymEnvs"

sbatch \
  -n 8 \
  --mem-per-cpu=8192 \
  -G 1 \
  --gres=gpumem:204800 \
  --time=04:00:00 \
  -A es_hutter \
  --mail-type=END \
  --tmp=15G \
  run.sh 
