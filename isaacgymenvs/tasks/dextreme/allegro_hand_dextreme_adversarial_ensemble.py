# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#	list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#	this list of conditions and the following disclaimer in the documentation
#	and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#	contributors may be used to endorse or promote products derived from
#	this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import math 
import os

from typing import Tuple, Dict, List, Set

import numpy as np

import torch
from torch import Tensor

from isaacgymenvs.tasks.dextreme.adr_vec_task import ADRVecTask
from isaacgymenvs.tasks.dextreme.allegro_hand_dextreme_adversarial import AllegroHandDextremeAdversarialObservationsAndActions, BaseControllerPlugin
from isaacgymenvs.tasks.dextreme.allegro_hand_dextreme_finetuning import AdversarialActionAndObservationNoiseGeneratorPlugin
from isaacgymenvs.utils.torch_jit_utils import scale, unscale, tensor_clamp  
from isaacgymenvs.utils.torch_jit_utils import quat_from_euler_xyz, quat_mul, quat_conjugate

from gym import spaces
from isaacgym import gymtorch
import onnxruntime as ort

import matplotlib.pyplot as plt

debug = False

class AllegroHandDextremeAdversarialObservationsAndActionsEnsemble(AllegroHandDextremeAdversarialObservationsAndActions):
	def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
		super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

		self.load_ensemble_models()

	def _read_cfg(self):
		super()._read_cfg()
		self.ensemble_models_dir = os.path.join(self.cfg["ensemble_dir"], self.cfg["experiment"])
		os.makedirs(self.ensemble_models_dir, exist_ok=True)
		
	def load_ensemble_models(self):
		self.ensemble_models = []

		print("=====================================================================\n\n\n\n")
		for model in os.scandir(self.ensemble_models_dir):
			model_path = os.path.join(self.ensemble_models_dir, model)
			print(model, "\n\t", model_path)
			
			onnx_model = AdversarialActionAndObservationNoiseGeneratorPlugin(model_path, self.device)
			self.ensemble_models.append(onnx_model)

		self.num_ensemble_models = len(self.ensemble_models)
		print("num_ensemble_models: ", self.num_ensemble_models)
		print("\n\n\n\n=====================================================================")

@torch.jit.script
def compute_hand_reward(
	rew_buf, reset_buf, reset_goal_buf, progress_buf, hold_count_buf, cur_targets, prev_targets, hand_dof_vel, successes, consecutive_successes,
	max_episode_length: float, object_pos, object_rot, target_pos, target_rot,
	dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
	actions, action_penalty_scale: float, action_delta_penalty_scale: float, #max_velocity: float,
	action_noise, action_noise_penalty_scale: float,
	cube_rot_noise, cube_rot_noise_penalty_scale: float,
	cube_pos_noise, cube_pos_noise_penalty_scale: float,
	dof_pos_noise, dof_pos_noise_penalty_scale: float,
	success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
	fall_penalty: float, max_consecutive_successes: int, av_factor: float, num_success_hold_steps: int
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
	# Distance from the hand to the object
	goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)

	# Orientation alignment for the cube in hand and goal cube
	quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
	rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

	dist_rew = goal_dist * dist_reward_scale
	rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

	action_penalty =  action_penalty_scale * torch.sum(actions ** 2, dim=-1)
	action_delta_penalty = action_delta_penalty_scale * torch.sum((cur_targets - prev_targets) ** 2, dim=-1)
  
	action_noise_penalty = action_noise_penalty_scale * torch.sum(action_noise ** 4, dim=-1)
	cube_rot_noise_penalty = cube_rot_noise_penalty_scale * torch.sum(cube_rot_noise ** 4, dim=-1)
	cube_pos_noise_penalty = cube_pos_noise_penalty_scale * torch.sum(cube_pos_noise ** 4, dim=-1)
	dof_pos_noise_penalty = dof_pos_noise_penalty_scale * torch.sum(dof_pos_noise ** 4, dim=-1)

	max_velocity = 5.0 #rad/s
	vel_tolerance = 1.0
	velocity_penalty_coef = -0.05

	# todo add actions regularization

	velocity_penalty = velocity_penalty_coef * torch.sum((hand_dof_vel/(max_velocity - vel_tolerance)) ** 2, dim=-1)

	# Find out which envs hit the goal and update successes count
	goal_reached = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
	hold_count_buf = torch.where(goal_reached, hold_count_buf + 1, torch.zeros_like(goal_reached))

	goal_resets = torch.where(hold_count_buf > num_success_hold_steps, torch.ones_like(reset_goal_buf), reset_goal_buf)
	successes = successes + goal_resets

	# Success bonus: orientation is within `success_tolerance` of goal orientation
	reach_goal_rew = (goal_resets == 1) * reach_goal_bonus

	# Fall penalty: distance to the goal is larger than a threashold
	fall_rew = (goal_dist >= fall_dist) * fall_penalty

	# Check env termination conditions, including maximum success number
	resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
	if max_consecutive_successes > 0:
		# Reset progress buffer on goal envs if max_consecutive_successes > 0
		progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
		resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)

	timed_out = progress_buf >= max_episode_length - 1
	resets = torch.where(timed_out, torch.ones_like(resets), resets)

	# Apply penalty for not reaching the goal
	timeout_rew = timed_out * 0.5 * fall_penalty

	# TODO:add KL Divergence term

	# Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
	reward = dist_rew + rot_rew + action_penalty + action_delta_penalty + velocity_penalty + reach_goal_rew + fall_rew + timeout_rew + \
			 action_noise_penalty + cube_pos_noise_penalty + cube_rot_noise_penalty + dof_pos_noise_penalty

	num_resets = torch.sum(resets)
	finished_cons_successes = torch.sum(successes * resets.float())

	cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

	return reward, resets, goal_resets, progress_buf, hold_count_buf, successes, cons_successes, \
		dist_rew, rot_rew, action_penalty, action_delta_penalty, velocity_penalty, reach_goal_rew, fall_rew, timeout_rew  # return individual rewards for visualization
