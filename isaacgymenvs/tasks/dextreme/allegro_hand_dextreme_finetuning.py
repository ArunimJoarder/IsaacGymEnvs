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

from isaacgymenvs.tasks.dextreme.adr_vec_task import ADRVecTask
from isaacgymenvs.tasks.dextreme.allegro_hand_dextreme import AllegroHandDextremeADR, AllegroHandDextreme, compute_hand_reward
from isaacgymenvs.tasks.dextreme.allegro_hand_dextreme_adversarial import BaseControllerPlugin
from isaacgymenvs.utils.torch_jit_utils import scale, unscale, tensor_clamp  
from isaacgymenvs.utils.torch_jit_utils import quat_from_euler_xyz, quat_mul, quat_conjugate

from gym import spaces
from isaacgym import gymtorch
import onnxruntime as ort

import matplotlib.pyplot as plt
import pickle

debug = False

class BaseNoiseGeneratorPlugin:
	def __init__(self, onnx_model_checkpoint, device) -> None:
		sess_options = ort.SessionOptions()
		# sess_options.inter_op_num_threads = 8
		# sess_options.intra_op_num_threads = 8
		# sess_options.log_severity_level = 0
		self._model = ort.InferenceSession(onnx_model_checkpoint, sess_options=sess_options, providers=["CUDAExecutionProvider"])
		if debug: print("[TASK-Finetuning][DEBUG] ONNX model input names:", [o.name for o in self._model.get_inputs()])
		self.device = device
		self.obs_spec = {
							'obs': {
								'names': ['dof_pos',
										  'dof_vel',
										  'dof_force',
										  'object_pose',
										  'object_pose_cam_randomized',
										  'object_vels',
										  'goal_pose',
										  'goal_relative_rot',
										  'last_actions',
										  'last_actions_full',
										  'stochastic_delay_params',
										  'affine_params',
										  'cube_random_params',
										  'hand_random_params',
										  'ft_force_torques',
										  'gravity_vec',
										  'ft_states',
										  'rot_dist',
										  'rb_forces'],
								'concat': True,
								'space_name': 'observation_space'
							},
							'states': {
								'names': ['dof_pos',
										  'dof_vel',
										  'dof_force',
										  'object_pose',
										  'object_pose_cam_randomized',
										  'object_vels',
										  'goal_pose',
										  'goal_relative_rot',
										  'last_actions',
										  'stochastic_delay_params',
										  'affine_params',
										  'cube_random_params',
										  'hand_random_params',
										  'ft_force_torques',
										  'gravity_vec',
										  'ft_states',
										  'rot_dist',
										  'rb_forces'], 
								'concat': True, 
								'space_name': 'state_space'
							}
						}

		self.action_noise_scale = 0.01
		self.cube_pos_noise_scale = 0.005
		self.cube_rot_noise_scale = 0.05
		self.dof_pos_noise_scale = 0.01
		self.num_rnn_states = 256

		self.init_rnn_needed = True

	def _generate_obs(self, env_obs: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
		rlgames_obs = {k: self.gen_obs_dict(env_obs, v['names'], v['concat']) for k, v in self.obs_spec.items()}
		return rlgames_obs
	
	def gen_obs_dict(self, obs_dict, obs_names, concat):
		if concat:
			# print(obs_dict)
			return torch.cat([obs_dict[name] for name in obs_names], dim=1)
		else:
			return {k: obs_dict[k] for k in obs_names}
			
	def obs_to_numpy(self, obs):
		obs = obs["obs"].cpu().numpy().astype(np.float32)
		return obs

	def init_rnn(self, batch_size):
		self.states =   (np.zeros((1, batch_size, self.num_rnn_states), dtype=np.float32), 
						 np.zeros((1, batch_size, self.num_rnn_states), dtype=np.float32))

	def rescale_noises(self, low, high, noise):
		d = (high - low) / 2.0
		m = (high + low) / 2.0
		scaled_noise =  noise * d + m
		return scaled_noise

	# NOTE: Function definition is specific to AllegroHandDextremeADR trained model 
	def get_noise(self, obs):
		obs = self._generate_obs(obs)
		np_obs = self.obs_to_numpy(obs)
		if self.init_rnn_needed:
			batch_size = np_obs.shape[0]
			self.init_rnn(batch_size)
			self.init_rnn_needed = False
		
		if debug: 
			print("[TASK-Finetuning][DEBUG] Obs shape", np_obs.shape, type(np_obs))
			for i,e in enumerate(self.states):
				print(f"[TASK-Finetuning][DEBUG] State {i} shape", e.shape, type(e))

		input_dict = {
			'obs' : np_obs,
			'out_state.1' : self.states[0],
			'hidden_state.1' : self.states[1],
		}
		
		if debug: print("[TASK-Finetuning][DEBUG] Running ONNX model")
		
		mu, out_states, hidden_states = self._model.run(None, input_dict)
		
		if debug: 
			print("[TASK-Finetuning][DEBUG] ONNX model ran successfully")
			print(f"[TASK-Finetuning][DEBUG] mu shape:", mu.shape)
			print(f"[TASK-Finetuning][DEBUG] out_states shape:", out_states.shape)
			print(f"[TASK-Finetuning][DEBUG] hidden_states shape:", hidden_states.shape)
		
		self.states = (out_states, hidden_states)
		current_noise = torch.tensor(mu).to(self.device)
		
		# TODO: Change hardcoded action high and low
		return self.rescale_noises(-1.0, 1.0, torch.clamp(current_noise, -1.0, 1.0))

class AdversarialActionNoiseGeneratorPlugin(BaseNoiseGeneratorPlugin):
	def __init__(self, onnx_model_checkpoint, device) -> None:
		super().__init__(onnx_model_checkpoint, device)
		self.obs_spec["obs"] = {'names': ['dof_pos',
										  'object_pose',
										  'goal_pose',
										  'goal_relative_rot',
										  'last_actions'],
								'concat': True,
								'space_name': 'observation_space'
								}

		self.num_rnn_states = 256

	def get_noise(self, obs):
		base_noise = super().get_noise(obs)

		noise = {}
		noise["action_noise"] = base_noise * self.action_noise_scale

		return noise, base_noise

class AdversarialActionAndObservationNoiseGeneratorPlugin(BaseNoiseGeneratorPlugin):
	def __init__(self, onnx_model_checkpoint, device) -> None:
		super().__init__(onnx_model_checkpoint, device)

		self.num_rnn_states = 256

	def get_noise(self, obs):
		base_noise = super().get_noise(obs)

		noise = {}
		noise["action_noise"] = base_noise[:,0:16] * self.action_noise_scale
		noise["cube_pos_noise"] = base_noise[:,16:19] * self.cube_pos_noise_scale
		noise["cube_rot_noise"] = base_noise[:,19:22] * self.cube_rot_noise_scale
		noise["dof_pos_noise"] = base_noise[:,22:38] * self.dof_pos_noise_scale

		return noise, base_noise

class AllegroHandDextremeADRFinetuning(AllegroHandDextremeADR):
	def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
		super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
		

	def _read_cfg(self):
		super()._read_cfg()
	
		self.noise_generator_checkpoint = self.cfg["onnx_noise_gen_checkpoint"]
		self.adv_noise_prob = self.cfg["env"]["adv_noise_prob"]

	def get_num_obs_dict(self, num_dofs):
		num_obs_dict = AllegroHandDextremeADR.get_num_obs_dict(self, num_dofs)
		num_obs_dict["last_actions_full"] = 16 + 3 + 3 + 16
		return num_obs_dict

	def _init_post_sim_buffers(self):
		super()._init_post_sim_buffers()
		self.noise_generator = AdversarialActionAndObservationNoiseGeneratorPlugin(self.noise_generator_checkpoint, self.device)

	def pre_physics_step(self, actions):
		self.use_adv_noise = False
		if np.random.rand() < self.adv_noise_prob:
			# print("============================= Adversarial Noise Used!!!!! =============================")
			self.use_adv_noise = True
		# else:
		# 	print("=======================================================================================")

		self.noises, self.noises_full = self.noise_generator.get_noise(self.obs_dict)
		
		if self.use_adv_noise and "action_noise" in self.noises.keys():
			# print("=============================   Action Noise Added!!!!!   =============================")
			actions = actions + self.noises["action_noise"]
	
		super().pre_physics_step(actions)

	def compute_observations(self):
		super().compute_observations()
		self.obs_dict["last_actions_full"][:] = self.noises_full

		if self.use_adv_noise and "cube_pos_noise" in self.noises.keys() and "cube_rot_noise" in self.noises.keys():
			noisy_cube_pos = self.obs_dict["object_pose_cam_randomized"][:, 0:3] + self.noises["cube_pos_noise"]

			cube_rot_noise_quat = quat_from_euler_xyz(self.noises["cube_rot_noise"][:, 0], 
													  self.noises["cube_rot_noise"][:, 1],
													  self.noises["cube_rot_noise"][:, 2])
			
			cube_rot = self.obs_dict["object_pose_cam_randomized"][:, 3:7]
			noisy_cube_rot = quat_mul(cube_rot, cube_rot_noise_quat)

			quat_diff = quat_mul(cube_rot, quat_conjugate(noisy_cube_rot))
			rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

			noisy_cube_pose_obs = torch.cat((noisy_cube_pos, noisy_cube_rot), axis=-1)
			
			# print("============================= Cube Pose Noise Added!!!!!  =============================")
			self.obs_dict["object_pose_cam_randomized"] = noisy_cube_pose_obs
			
		if self.use_adv_noise and "dof_pos_noise" in self.noises.keys():
			# print("=============================  DoF Pos Noise Added!!!!!   =============================")
			self.obs_dict["dof_pos_randomized"] = self.obs_dict["dof_pos_randomized"] + self.noises["dof_pos_noise"]

class AllegroHandDextremeADRFinetuningResidualActions(AllegroHandDextremeADR):
	def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
		super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

		self.delta_actions = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
		self.prev_delta_actions = torch.zeros_like(self.delta_actions, device=self.device)

		if not cfg["test"]:
			with open(cfg["adr_params_file"], "rb") as fp:
				self.adr_params = pickle.load(fp)
			print("Loaded adr_params from checkpoint!")

		self.realtime_plots = self.cfg["plot_delta_actions"]
		if self.realtime_plots:
			if self.num_envs <= 4:
				self.rt_plot_fig_actions, self.rt_plot_ax_actions = plt.subplots(4,4, sharey=True, sharex=True)

				self.rt_plot_fig_actions.suptitle("Residual Actions")
				for i, ax in enumerate(self.rt_plot_ax_actions.flat):
					ax.set(xlabel='frame')
					ax.set_title(f"dof_{i+1} Delta Action")

				self.rt_plot_fig_actions.show()
				# plt.pause(0.0001)

				self.rt_plot_actions_buffer = np.zeros((1, self.num_envs, 16))
				self.rt_plot_frame_buffer = np.zeros(1)
			else:
				self.realtime_plots = False
				print("Real-time plots cannot be plotted with more than 4 agents being simulated at once.") 

	def _read_cfg(self):
		super()._read_cfg()
	
		self.base_controller_checkpoint = self.cfg["onnx_model_checkpoint"]
		self.noise_generator_checkpoint = self.cfg["onnx_noise_gen_checkpoint"]
		self.adv_noise_prob = self.cfg["env"]["adv_noise_prob"]

	def get_num_obs_dict(self, num_dofs):
		num_obs_dict = AllegroHandDextremeADR.get_num_obs_dict(self, num_dofs)
		num_obs_dict["last_actions_full"] = 16 + 3 + 3 + 16
		num_obs_dict["base_actions"] = 16
		return num_obs_dict

	def _init_post_sim_buffers(self):
		super()._init_post_sim_buffers()
		self.noise_generator = AdversarialActionAndObservationNoiseGeneratorPlugin(self.noise_generator_checkpoint, self.device)
		self.base_controller = BaseControllerPlugin(self.base_controller_checkpoint, self.device)

	def pre_physics_step(self, delta_actions):
		self.use_adv_noise = False
		if np.random.rand() < self.adv_noise_prob:
			self.use_adv_noise = True

		self.noises, self.noises_full = self.noise_generator.get_noise(self.obs_dict)
		
		self.base_actions = self.base_controller.get_action(self.obs_dict)

		self.prev_delta_actions = self.delta_actions
		self.delta_actions = delta_actions
		# self.delta_actions = 2.0 * delta_actions

		actions = self.base_actions + self.delta_actions
		actions = torch.clamp(actions, -1.0, 1.0)
		if self.use_adv_noise and "action_noise" in self.noises.keys():
			actions = actions + self.noises["action_noise"]
	
		super().pre_physics_step(actions)

	def compute_observations(self):
		super().compute_observations()
		self.obs_dict["last_actions_full"][:] = self.noises_full
		self.obs_dict["base_actions"][:] = self.base_actions

		if self.use_adv_noise and "cube_pos_noise" in self.noises.keys() and "cube_rot_noise" in self.noises.keys():
			noisy_cube_pos = self.obs_dict["object_pose_cam_randomized"][:, 0:3] + self.noises["cube_pos_noise"]

			cube_rot_noise_quat = quat_from_euler_xyz(self.noises["cube_rot_noise"][:, 0], 
													  self.noises["cube_rot_noise"][:, 1],
													  self.noises["cube_rot_noise"][:, 2])
			
			cube_rot = self.obs_dict["object_pose_cam_randomized"][:, 3:7]
			noisy_cube_rot = quat_mul(cube_rot, cube_rot_noise_quat)

			quat_diff = quat_mul(cube_rot, quat_conjugate(noisy_cube_rot))
			rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

			noisy_cube_pose_obs = torch.cat((noisy_cube_pos, noisy_cube_rot), axis=-1)
			
			self.obs_dict["object_pose_cam_randomized"] = noisy_cube_pose_obs
			
		if self.use_adv_noise and "dof_pos_noise" in self.noises.keys():
			self.obs_dict["dof_pos_randomized"] = self.obs_dict["dof_pos_randomized"] + self.noises["dof_pos_noise"]

	def plot_residual_actions(self):
		if self.realtime_plots:
			if self.frame % 10 == 0:
				for ax in self.rt_plot_ax_actions.flat:
					ax.cla()
				self.rt_plot_frame_buffer = np.vstack([self.rt_plot_frame_buffer, self.frame])
				self.rt_plot_actions_buffer = np.vstack([self.rt_plot_actions_buffer, np.array([self.delta_actions.cpu().numpy(),])])

				for i in range(self.num_envs):
					for j, ax in enumerate(self.rt_plot_ax_actions.flat):
						# ax.plot(self.rt_plot_frame_buffer[:], self.rt_plot_actions_buffer[:, i, j])
						ax.plot(self.rt_plot_frame_buffer[-min(50, self.rt_plot_frame_buffer.shape[0]):-1], self.rt_plot_actions_buffer[-min(50, self.rt_plot_frame_buffer.shape[0]):-1, i, j])
				for i, ax in enumerate(self.rt_plot_ax_actions.flat):
					# ax.set(xlabel='frame')
					ax.set_title(f"dof_{i+1} Delta Action")
				plt.pause(0.00000000001)

	def compute_reward(self, actions):
		self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], \
		self.hold_count_buf[:], self.successes[:], self.consecutive_successes[:], \
		dist_rew, rot_rew, action_penalty, action_delta_penalty, velocity_penalty, reach_goal_rew, fall_rew, timeout_rew = compute_hand_reward(
			self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.hold_count_buf, self.delta_actions, self.prev_delta_actions,
			self.dof_vel, self.successes, self.consecutive_successes, self.max_episode_length,
			self.object_pos, self.object_rot, self.goal_pos, self.goal_rot, self.dist_reward_scale, self.rot_reward_scale, self.rot_eps,
			self.delta_actions, self.action_penalty_scale, self.action_delta_penalty_scale,
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
		episode_cumulative['action_penalty'] = action_penalty
		episode_cumulative['action_delta_penalty'] = action_delta_penalty
		episode_cumulative['velocity_penalty'] = velocity_penalty
		episode_cumulative['reach_goal_rew'] = reach_goal_rew
		episode_cumulative['fall_rew'] = fall_rew
		episode_cumulative['timeout_rew'] = timeout_rew
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

				# actions_abs_avg = torch.abs(self.actions).mean(dim=0)
				# # print(self.actions.size())
				# # print(actions_abs_avg.size())
				# for i in range(16):
				#	 print(f"action_abs_avg/dof_{i+1} = ", actions_abs_avg[i].item())
				# # print(self.actions.cpu().numpy())

				# create a matplotlib bar chart of the self.successes_count
				import matplotlib.pyplot as plt
				plt.bar(list(range(self.max_consecutive_successes + 1)), self.successes_count.cpu().numpy())
				plt.title("Successes histogram")
				plt.xlabel("Successes")
				plt.ylabel("Frequency")
				plt.savefig(f"{self.eval_summary_dir}/successes_histogram.png")
				plt.clf()

		self.plot_residual_actions()
