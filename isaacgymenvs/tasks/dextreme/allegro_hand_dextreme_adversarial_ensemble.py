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
from isaacgymenvs.tasks.dextreme.allegro_hand_dextreme_adversarial import AllegroHandDextremeAdversarialObservationsAndActions
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

		self.ensemble_models = []
		self.load_ensemble_models()
		self.ensemble_noises = []
		self.ensemble_mus = []
		self.ensemble_sigmas = []
		for i in range(self.num_ensemble_models):
			self.ensemble_noises.append(torch.zeros((self.num_envs, self.num_actions), dtype=torch.float, device=self.device))
			self.ensemble_mus.append(torch.zeros((self.num_envs, self.num_actions), dtype=torch.float, device=self.device))
			self.ensemble_sigmas.append(torch.zeros((self.num_envs, self.num_actions), dtype=torch.float, device=self.device))

	def _read_cfg(self):
		super()._read_cfg()
		self.ensemble_models_dir = os.path.join(self.cfg["ensemble_dir"], self.cfg["experiment"])
		os.makedirs(self.ensemble_models_dir, exist_ok=True)

		self.kl_penalty_scale = self.cfg["env"]["klPenaltyScale"]

	def load_ensemble_models(self):
		for model in os.scandir(self.ensemble_models_dir):
			model_path = os.path.join(self.ensemble_models_dir, model)
			print("ensemble_checkpoint: ", model_path)
			
			onnx_model = AdversarialActionAndObservationNoiseGeneratorPlugin(model_path, self.device)
			self.ensemble_models.append(onnx_model)

		self.num_ensemble_models = len(self.ensemble_models)
		print("num_ensemble_models: ", self.num_ensemble_models)
		
	def pre_physics_step(self, obs_and_action_noises):
		for i in range(self.num_ensemble_models):
			_, self.ensemble_noises[i], res_dict = self.ensemble_models[i].get_noise(self.obs_dict)
			self.ensemble_mus[i] = torch.from_numpy(res_dict["mus"]).to(self.device)
			self.ensemble_sigmas[i] = torch.from_numpy(res_dict["sigmas"]).to(self.device)

		super().pre_physics_step(obs_and_action_noises)
	
	def compute_reward(self, actions):

		self.mus = self.res_dict["mus"]
		self.sigmas = self.res_dict["sigmas"]

		self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], \
		self.hold_count_buf[:], self.successes[:], self.consecutive_successes[:], \
		dist_rew, rot_rew, action_penalty, action_delta_penalty, velocity_penalty, reach_goal_rew, fall_rew, timeout_rew, \
		action_noise_penalty, cube_rot_noise_penalty, cube_pos_noise_penalty, dof_pos_noise_penalty, kl_penalty = compute_hand_reward(
			self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.hold_count_buf, self.cur_targets, self.prev_targets,
			self.dof_vel, self.successes, self.consecutive_successes, self.max_episode_length,
			self.object_pos, self.object_rot, self.goal_pos, self.goal_rot, self.dist_reward_scale, self.rot_reward_scale, self.rot_eps,
			self.actions, self.action_penalty_scale, self.action_delta_penalty_scale,
			self.action_noise, self.action_noise_penalty_scale,
			self.cube_rot_noise, self.cube_rot_noise_penalty_scale,
			self.cube_pos_noise, self.cube_pos_noise_penalty_scale,
			self.dof_pos_noise, self.dof_pos_noise_penalty_scale,
			self.mus, self.sigmas, self.ensemble_mus, self.ensemble_sigmas, self.kl_penalty_scale,
			self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
			self.max_consecutive_successes, self.av_factor, self.num_success_hold_steps
		)

		# update best rotation distance in the current episode
		self.best_rotation_dist = torch.minimum(self.best_rotation_dist, self.curr_rotation_dist)
		self.extras['consecutive_successes'] = self.consecutive_successes.mean()
		self.extras['true_objective'] = self.successes

		episode_cumulative = dict()
		episode_cumulative['dist_rew'] = dist_rew
		episode_cumulative['rot_rew'] = rot_rew
		# episode_cumulative['action_penalty'] = action_penalty
		# episode_cumulative['action_delta_penalty'] = action_delta_penalty
		episode_cumulative['velocity_penalty'] = velocity_penalty
		episode_cumulative['reach_goal_rew'] = reach_goal_rew
		episode_cumulative['fall_rew'] = fall_rew
		episode_cumulative['timeout_rew'] = timeout_rew
		episode_cumulative['action_noise_penalty'] = action_noise_penalty
		episode_cumulative['cube_rot_noise_penalty'] = cube_rot_noise_penalty
		episode_cumulative['cube_pos_noise_penalty'] = cube_pos_noise_penalty
		episode_cumulative['dof_pos_noise_penalty'] = dof_pos_noise_penalty
		episode_cumulative['kl_penalty'] = kl_penalty
		self.extras['episode_cumulative'] = episode_cumulative

		if self.print_success_stat:
			is_success = self.reset_goal_buf.to(torch.bool)

			frame_ = torch.empty_like(self.last_success_step).fill_(self.frame)
			self.success_time = torch.where(is_success, frame_ - self.last_success_step, self.success_time)
			self.last_success_step = torch.where(is_success, frame_, self.last_success_step)
			mask_ = self.success_time > 0
			if any(mask_):
				avg_time_mean = ((self.success_time * mask_).sum(dim=0) / mask_.sum(dim=0)).item()
			else:
				avg_time_mean = math.nan
			
			envs_reset = self.reset_buf 
			if self.use_adr:
				envs_reset = self.reset_buf & ~self.apply_reset_buf
			
			self.total_resets = self.total_resets + envs_reset.sum() 
			direct_average_successes = self.total_successes + self.successes.sum()
			self.total_successes = self.total_successes + (self.successes * envs_reset).sum() 

			self.total_num_resets += envs_reset

			self.last_ep_successes = torch.where(envs_reset > 0, self.successes, self.last_ep_successes)
			reset_ids = envs_reset.nonzero().squeeze()
			last_successes = self.successes[reset_ids].long()
			self.successes_count[last_successes] += 1

			if self.frame % 100 == 0:
				# The direct average shows the overall result more quickly, but slightly undershoots long term
				# policy performance.
				print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
				if self.total_resets > 0:
					print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))
				print(f"Max num successes: {self.successes.max().item()}")
				print(f"Average consecutive successes: {self.consecutive_successes.mean().item():.2f}")
				print(f"Total num resets: {self.total_num_resets.sum().item()} --> {self.total_num_resets}")
				print(f"Reset percentage: {(self.total_num_resets > 0).sum() / self.num_envs:.2%}")

				print(f"Last ep successes: {self.last_ep_successes.mean().item():.2f} {self.last_ep_successes}")

				self.eval_summaries.add_scalar("consecutive_successes", self.consecutive_successes.mean().item(), self.frame)
				self.eval_summaries.add_scalar("last_ep_successes", self.last_ep_successes.mean().item(), self.frame)
				self.eval_summaries.add_scalar("reset_stats/reset_percentage", (self.total_num_resets > 0).sum() / self.num_envs, self.frame)
				self.eval_summaries.add_scalar("reset_stats/min_num_resets", self.total_num_resets.min().item(), self.frame)

				self.eval_summaries.add_scalar("policy_speed/avg_success_time_frames", avg_time_mean, self.frame)
				frame_time = self.control_freq_inv * self.dt
				self.eval_summaries.add_scalar("policy_speed/avg_success_time_seconds", avg_time_mean * frame_time, self.frame)
				self.eval_summaries.add_scalar("policy_speed/avg_success_per_minute", 60.0 / (avg_time_mean * frame_time), self.frame)
				print(f"Policy speed (successes per minute): {60.0 / (avg_time_mean * frame_time):.2f}")

				dof_delta = self.dof_delta.abs()
				print(f"Max dof deltas: {dof_delta.max(dim=0).values}, max across dofs: {self.dof_delta.abs().max().item():.2f}, mean: {self.dof_delta.abs().mean().item():.2f}")
				print(f"Max dof delta radians per sec: {dof_delta.max().item() / frame_time:.2f}, mean: {dof_delta.mean().item() / frame_time:.2f}")

				dof_pos_noise_avg = self.dof_pos_noise_scaled.mean(dim=0)
				object_rot_noise_avg = self.cube_rot_noise_scaled.mean(dim=0)
				object_pos_noise_avg = self.cube_pos_noise_scaled.mean(dim=0)
				action_noise_avg = self.action_noise_scaled.mean(dim=0)
				object_rot_noise_avg_angle = self.object_rot_noise.mean()

				labels = ["x", "y", "z"]
				for i in range(3):
					self.eval_summaries.add_scalar(f"object_pose_noise_avg/pos_{labels[i]}", object_pos_noise_avg[i].item(), self.frame)
					self.eval_summaries.add_scalar(f"object_pose_noise_avg/rot_{labels[i]}", object_rot_noise_avg[i].item(), self.frame)

				for i in range(16):
					self.eval_summaries.add_scalar(f"dof_pos_noise_avg/dof_{i+1}", dof_pos_noise_avg[i].item(), self.frame)
					self.eval_summaries.add_scalar(f"action_noise_avg/dof_{i+1}", action_noise_avg[i].item(), self.frame)

				# create a matplotlib bar chart of the self.successes_count
				# import matplotlib.pyplot as plt
				plt.bar(list(range(self.max_consecutive_successes + 1)), self.successes_count.cpu().numpy())
				plt.title("Successes histogram")
				plt.xlabel("Successes")
				plt.ylabel("Frequency")
				plt.savefig(f"{self.eval_summary_dir}/successes_histogram.png")
				plt.clf()

		if self.realtime_plots:
			self.realtime_plotter()

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
	mus, sigmas, ensemble_mus: List[Tensor], ensemble_sigmas: List[Tensor], kl_penalty_scale: float,
	success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
	fall_penalty: float, max_consecutive_successes: int, av_factor: float, num_success_hold_steps: int
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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
	kl_penalty = torch.zeros_like(timeout_rew)
	for i in range(len(ensemble_sigmas)):
		c1 = torch.log(sigmas/ensemble_sigmas[i] + 1e-5)
		c2 = (ensemble_sigmas[i]**2 + (mus - ensemble_mus[i])**2)/(2.0 * (sigmas**2 + 1e-5))
		c3 = -1.0 / 2.0
		kl = c1 + c2 + c3
		kl = kl.sum(dim=-1)

		kl_penalty += kl
	kl_penalty *= kl_penalty_scale

	# Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
	reward = dist_rew + rot_rew + action_penalty + action_delta_penalty + velocity_penalty + reach_goal_rew + fall_rew + timeout_rew + \
			 action_noise_penalty + cube_pos_noise_penalty + cube_rot_noise_penalty + dof_pos_noise_penalty + kl_penalty

	num_resets = torch.sum(resets)
	finished_cons_successes = torch.sum(successes * resets.float())

	cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

	return reward, resets, goal_resets, progress_buf, hold_count_buf, successes, cons_successes, \
		dist_rew, rot_rew, action_penalty, action_delta_penalty, velocity_penalty, reach_goal_rew, fall_rew, timeout_rew, action_noise_penalty, cube_rot_noise_penalty, cube_pos_noise_penalty, dof_pos_noise_penalty, kl_penalty  # return individual rewards for visualization
