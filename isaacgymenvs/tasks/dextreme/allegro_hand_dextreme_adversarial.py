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
from isaacgymenvs.tasks.dextreme.allegro_hand_dextreme import AllegroHandDextremeADR, AllegroHandDextreme
from isaacgymenvs.utils.torch_jit_utils import scale, unscale, tensor_clamp  

from gym import spaces
from isaacgym import gymtorch
import onnxruntime as ort

debug = False

class BaseControllerPlugin:
	def __init__(self, onnx_model_checkpoint, device) -> None:
		self._model = ort.InferenceSession(onnx_model_checkpoint, providers=["CUDAExecutionProvider"])
		if debug: print("[TASK-AdvActions][DEBUG] ONNX model input names:", [o.name for o in self._model.get_inputs()])
		self.device = device
		self.obs_spec = {
							'obs': {
								'names': ['dof_pos_randomized',
										  'object_pose_cam_randomized',
										  'goal_pose',
										  'goal_relative_rot_cam_randomized',
										  'last_actions'],
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
		self.states =   (np.zeros((1, batch_size, 1024), dtype=np.float32), 
						 np.zeros((1, batch_size, 1024), dtype=np.float32))

	def rescale_actions(self, low, high, action):
		d = (high - low) / 2.0
		m = (high + low) / 2.0
		scaled_action =  action * d + m
		return scaled_action

	# NOTE: Function definition is specific to AllegroHandDextremeADR trained model 
	def get_action(self, obs):
		obs = self._generate_obs(obs)
		np_obs = self.obs_to_numpy(obs)
		if self.init_rnn_needed:
			batch_size = np_obs.shape[0]
			self.init_rnn(batch_size)
			self.init_rnn_needed = False
		
		if debug: 
			print("[TASK-AdvActions][DEBUG] Obs shape", np_obs.shape, type(np_obs))
			for i,e in enumerate(self.states):
				print(f"[TASK-AdvActions][DEBUG] State {i} shape", e.shape, type(e))

		input_dict = {
			'obs' : np_obs,
			'out_state.1' : self.states[0],
			'hidden_state.1' : self.states[1],
		}
		
		if debug: print("[TASK-AdvActions][DEBUG] Running ONNX model")
		
		mu, out_states, hidden_states = self._model.run(None, input_dict)
		
		if debug: 
			print("[TASK-AdvActions][DEBUG] ONNX model ran successfully")
			print(f"[TASK-AdvActions][DEBUG] mu shape:", mu.shape)
			print(f"[TASK-AdvActions][DEBUG] out_states shape:", out_states.shape)
			print(f"[TASK-AdvActions][DEBUG] hidden_states shape:", hidden_states.shape)
		
		self.states = (out_states, hidden_states)
		current_action = torch.tensor(mu).to(self.device)
		
		# TODO: Change hardcoded action high and low
		return self.rescale_actions(-1.0, 1.0, torch.clamp(current_action, -1.0, 1.0))

class AllegroHandDextremeAdversarialActions(AllegroHandDextremeADR):
	def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
		super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
		
		# Make sure output from network is within -0.1, 0.1
		self.clip_actions = self.cfg["env"].get("clipActions", 0.1)		
		self.act_space = spaces.Box(np.ones(self.num_actions) * -self.clip_actions, np.ones(self.num_actions) * self.clip_actions)

	def _read_cfg(self):
		super()._read_cfg()
	
		self.base_controller_checkpoint = self.cfg["onnx_model_checkpoint"]

	def _init_post_sim_buffers(self):
		super()._init_post_sim_buffers()
		self.base_controller = BaseControllerPlugin(self.base_controller_checkpoint, self.device)

	def pre_physics_step(self, action_noises):
		# Anneal action moving average 
		self.update_action_moving_average()
	   
		env_ids_reset = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
		goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

		if self.randomize and not self.use_adr:
			self.apply_randomizations(dr_params=self.randomization_params, randomisation_callback=self.randomisation_callback)

		elif self.randomize and self.use_adr:
					   
			# NB - when we are daing ADR, we must calculate the ADR or new DR vals one step BEFORE applying randomisations
			# this is because reset needs to be applied on the next step for it to take effect
			env_mask_randomize = (self.reset_buf & ~self.apply_reset_buf).bool()
			env_ids_reset = self.apply_reset_buf.nonzero(as_tuple=False).flatten()			
			if len(env_mask_randomize.nonzero(as_tuple=False).flatten()) > 0:
				self.apply_randomizations(dr_params=self.randomization_params,
										 randomize_buf=env_mask_randomize,
										 adr_objective=self.successes,
										 randomisation_callback=self.randomisation_callback)

				self.apply_reset_buf[env_mask_randomize] = 1

		# if only goals need reset, then call set API
		if len(goal_env_ids) > 0 and len(env_ids_reset) == 0:
			self.reset_target_pose(goal_env_ids, apply_reset=True)

		# if goals need reset in addition to other envs, call set API in reset()
		elif len(goal_env_ids) > 0:
			self.reset_target_pose(goal_env_ids)

		if len(env_ids_reset) > 0:
			self.reset_idx(env_ids_reset, goal_env_ids)

		self.action_noises = action_noises
		self.base_controller_actions = self.base_controller.get_action(self.obs_dict)
		actions = self.base_controller_actions + self.action_noises

		self.apply_actions(actions)
		self.apply_random_forces()

	def compute_observations(self):
		super().compute_observations()

		self.obs_dict["last_actions"][:] = self.base_controller_actions

	def compute_reward(self, actions):
		super().compute_reward(actions)

		if self.print_success_stat:
			if self.frame % 100 == 0:
				last_action_noise_avg = self.action_noises.mean(dim=1)
				last_base_action_avg = self.base_controller_actions.mean(dim=1)

				for i in range(16):
					# self.eval_summaries.add_scalar(f"base_actions_avg/actuator_{i+1}", last_base_action_avg[i].item(), self.frame)
					self.eval_summaries.add_scalar(f"action_noises_avg/actuator_{i+1}", last_action_noise_avg[i].item(), self.frame)

class AllegroHandDextremeAdversarialObservationsAndActions(AllegroHandDextremeADR, AllegroHandDextreme, ADRVecTask):
	def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

		'''
		obligatory constructor to fill-in class variables and setting
		up the simulation.

		self._read_cfg() is about initialising class variables from a 
						 config file.
		
		self._init_pre_sim_buffers() initialises particular tensors 
						 that are useful in storing various states 
						 randomised or otherwise 

		self._init_post_sim_buffers() initialises the root tensors and
						 other auxiliary variables that can be provided
						 as input to the controller or the value function	 

		'''

		self.cfg = cfg

		# Read the task config file and store all the relevant variables in the class
		self._read_cfg()

		self.fingertips = [s+"_link_3" for s in ["index", "middle", "ring", "thumb"]]
		self.num_fingertips = len(self.fingertips)
		num_dofs = 16
		
		self.num_obs_dict = self.get_num_obs_dict(num_dofs)

		self.cfg["env"]["obsDims"] = {} 

		for o in self.num_obs_dict.keys():
			if o not in self.num_obs_dict:
				raise Exception(f"Unknown type of observation {o}!")
			self.cfg["env"]["obsDims"][o] = (self.num_obs_dict[o],)

		self.up_axis = 'z'

		self.use_vel_obs = False
		self.fingertip_obs = True
		self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

		self.cfg["env"]["numActions"] = 16 + 7 + 16

		self.sim_device = sim_device

		rl_device = self.cfg.get("rl_device", "cuda:0")

		self._init_pre_sim_buffers()
		ADRVecTask.__init__(self, config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, use_dict_obs=True)
		self._init_post_sim_buffers()

		reward_keys = ['dist_rew', 'rot_rew', 'action_penalty', 'action_delta_penalty',
					   'velocity_penalty', 'reach_goal_rew', 'fall_rew', 'timeout_rew']
		self.rewards_episode = {key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in reward_keys}

		if self.use_adr:
			self.apply_reset_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device) 


		if self.print_success_stat:						
			self.last_success_step = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
			self.success_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
			self.last_ep_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
			self.total_num_resets = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
			self.successes_count = torch.zeros(self.max_consecutive_successes + 1, dtype=torch.float, device=self.device)
			from tensorboardX import SummaryWriter
			self.eval_summary_dir = "./eval_summaries"
			if self.cfg["experiment_dir"] != '':
				self.eval_summary_dir += ("/" + self.cfg["experiment_dir"])
			if self.cfg["experiment"] != '':
				self.eval_summary_dir += ("/" + self.cfg["experiment"])

			# remove the old directory if it exists
			if os.path.exists(self.eval_summary_dir):
				import shutil
				shutil.rmtree(self.eval_summary_dir)
			self.eval_summaries = SummaryWriter(self.eval_summary_dir, flush_secs=3)

		# Make sure output from network is within -0.1, 0.1
		self.clip_action_noise = self.cfg["env"].get("clipActionNoise", 0.1)		
		self.clip_cube_pose_noise = self.cfg["env"].get("clipCubePoseNoise", 5.0)		
		self.clip_dof_pos_noise = self.cfg["env"].get("clipDofPosNoise", 0.1)		
		box_boundary = np.concatenate((np.ones(16) * self.clip_action_noise, np.ones(7) * self.clip_cube_pose_noise, np.ones(16) * self.clip_dof_pos_noise))
		self.act_space = spaces.Box(-box_boundary, box_boundary)

		# self.num_actions = 16

	def get_num_obs_dict(self, num_dofs):
		num_obs_dict = AllegroHandDextremeADR.get_num_obs_dict(self, num_dofs)
		num_obs_dict["last_actions_full"] = 16 + 7 + 16
		num_obs_dict["last_actions"] = 16
		return num_obs_dict

	def _read_cfg(self):
		AllegroHandDextremeADR._read_cfg(self)
		self.base_controller_checkpoint = self.cfg["onnx_model_checkpoint"]

	def _init_pre_sim_buffers(self):
		AllegroHandDextremeADR._init_pre_sim_buffers(self)
		self.affine_actions_additive = torch.zeros(self.cfg["env"]["numEnvs"], 16, dtype=torch.float, device=self.sim_device)
		self.affine_actions_scaling = torch.zeros(self.cfg["env"]["numEnvs"], 16, dtype=torch.float, device=self.sim_device)

	def _init_post_sim_buffers(self):
		AllegroHandDextremeADR._init_post_sim_buffers(self)
		self.prev_actions = torch.zeros(self.num_envs, 16, dtype=torch.float, device=self.device)
		self.prev_actions_queue = torch.zeros(self.cfg["env"]["numEnvs"], self.action_latency_max + 1, 16, dtype=torch.float, device=self.sim_device)
		
		self.base_controller = BaseControllerPlugin(self.base_controller_checkpoint, self.device)

	def pre_physics_step(self, obs_and_action_noises):
		# Anneal action moving average 
		self.update_action_moving_average()
	   
		env_ids_reset = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
		goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

		if self.randomize and not self.use_adr:
			self.apply_randomizations(dr_params=self.randomization_params, randomisation_callback=self.randomisation_callback)

		elif self.randomize and self.use_adr:
					   
			# NB - when we are daing ADR, we must calculate the ADR or new DR vals one step BEFORE applying randomisations
			# this is because reset needs to be applied on the next step for it to take effect
			env_mask_randomize = (self.reset_buf & ~self.apply_reset_buf).bool()
			env_ids_reset = self.apply_reset_buf.nonzero(as_tuple=False).flatten()			
			if len(env_mask_randomize.nonzero(as_tuple=False).flatten()) > 0:
				self.apply_randomizations(dr_params=self.randomization_params,
										 randomize_buf=env_mask_randomize,
										 adr_objective=self.successes,
										 randomisation_callback=self.randomisation_callback)

				self.apply_reset_buf[env_mask_randomize] = 1

		# if only goals need reset, then call set API
		if len(goal_env_ids) > 0 and len(env_ids_reset) == 0:
			self.reset_target_pose(goal_env_ids, apply_reset=True)

		# if goals need reset in addition to other envs, call set API in reset()
		elif len(goal_env_ids) > 0:
			self.reset_target_pose(goal_env_ids)

		if len(env_ids_reset) > 0:
			self.reset_idx(env_ids_reset, goal_env_ids)

		self.cube_pose_noise_scaled = obs_and_action_noises[:,16:23]
		self.dof_pos_noise_scaled = obs_and_action_noises[:,23:39]
		self.add_noise_to_observations()
		self.base_controller_actions = self.base_controller.get_action(self.obs_dict)
		# self.base_controller_actions = self.base_controller.get_action(self.obs_dict)
		self.action_noise_scaled = obs_and_action_noises[:,0:16]
		actions = self.base_controller_actions# + self.action_noise_scaled

		self.actions_full = obs_and_action_noises.clone().to(self.device)

		self.apply_actions(actions)
		self.apply_random_forces()

	def add_noise_to_observations(self):
		self.obs_dict["object_pose_cam_randomized"] = self.obs_dict["object_pose_cam_randomized"] + self.cube_pose_noise_scaled
		self.obs_dict["dof_pos_randomized"] = self.obs_dict["dof_pos_randomized"] + self.dof_pos_noise_scaled
	
	def apply_action_noise_latency(self):
		
		action_delay_mask = (torch.rand(self.num_envs, device=self.device) < self.get_adr_tensor("action_delay_prob")).view(-1, 1)			

		actions = \
				self.prev_actions_queue[torch.arange(self.prev_actions_queue.shape[0]), self.action_latency] * ~action_delay_mask \
				+ self.prev_actions * action_delay_mask
		
		white_noise = self.sample_gaussian_adr("affine_action_white", self.all_env_ids, trailing_dim=16)
		actions = self.affine_actions_scaling * actions + self.affine_actions_additive + white_noise
					
		return actions
	
	def apply_randomizations(self, dr_params, randomize_buf, adr_objective=None, randomisation_callback=None):

		AllegroHandDextreme.apply_randomizations(self, dr_params, randomize_buf, adr_objective, randomisation_callback=self.randomisation_callback)

		randomize_env_ids = randomize_buf.nonzero(as_tuple=False).squeeze(-1)

		self.action_latency[randomize_env_ids] = self.sample_discrete_adr("action_latency", randomize_env_ids)

		self.cube_pose_refresh_rate[randomize_env_ids] = self.sample_discrete_adr("cube_pose_refresh_rate", randomize_env_ids)

		# Nb - code is to generate uniform from 1 to max_skip_obs (inclusive), but cant use 
		# torch.uniform as it doesn't support a different max/min value on each
		self.cube_pose_refresh_offset[randomize_buf] = \
			(torch.rand(randomize_env_ids.shape, device=self.device, dtype=torch.float) \
				* (self.cube_pose_refresh_rate[randomize_env_ids].view(-1).float()) - 0.5).round().long() # offset range shifted back by one
		
		self.affine_actions_scaling[randomize_env_ids] = 1. + self.sample_gaussian_adr("affine_action_scaling", randomize_env_ids, trailing_dim=16)
		self.affine_actions_additive[randomize_env_ids] = self.sample_gaussian_adr("affine_action_additive", randomize_env_ids, trailing_dim=16)

		self.affine_cube_pose_scaling[randomize_env_ids] = 1. + self.sample_gaussian_adr("affine_cube_pose_scaling", randomize_env_ids, trailing_dim=7)
		self.affine_cube_pose_additive[randomize_env_ids] = self.sample_gaussian_adr("affine_cube_pose_additive", randomize_env_ids, trailing_dim=7)

		self.affine_dof_pos_scaling[randomize_env_ids] = 1. + self.sample_gaussian_adr("affine_dof_pos_scaling", randomize_env_ids, trailing_dim=16)
		self.affine_dof_pos_additive[randomize_env_ids] = self.sample_gaussian_adr("affine_dof_pos_additive", randomize_env_ids, trailing_dim=16)

	def apply_actions(self, actions):

		self.actions = actions.clone().to(self.device)

		refreshed = self.progress_buf == 0
		self.prev_actions_queue[refreshed] = unscale(self.dof_pos[refreshed], self.hand_dof_lower_limits, 
													 self.hand_dof_upper_limits).view(-1, 1, 16)
		
		# Needed for the first step and every refresh 
		# you don't want to mix with zeros
		self.prev_actions[refreshed] = unscale(self.dof_pos[refreshed], self.hand_dof_lower_limits, 
											   self.hand_dof_upper_limits).view(-1, 16)
		
		# update the actions queue
		self.prev_actions_queue[:, 1:] = self.prev_actions_queue[:, :-1].detach()
		self.prev_actions_queue[:, 0, :] = self.actions

		# apply action delay		 
		actions_delayed = self.apply_action_noise_latency()

		# apply random network adversary 
		actions_delayed = self.get_random_network_adversary_action(actions_delayed)

		if self.use_relative_control:

			targets = self.prev_targets[:, self.actuated_dof_indices] + self.hand_dof_speed_scale * self.dt * actions_delayed
			self.cur_targets[:, self.actuated_dof_indices]  = targets 

		elif self.use_capped_dof_control:

			# This is capping the maximum dof velocity
			targets = scale(actions_delayed, self.hand_dof_lower_limits[self.actuated_dof_indices], 
							self.hand_dof_upper_limits[self.actuated_dof_indices])

			delta = targets[:, self.actuated_dof_indices] - self.prev_targets[:, self.actuated_dof_indices]
			
			max_dof_delta = self.max_dof_radians_per_second * self.dt * self.control_freq_inv
			
			delta = torch.clamp_(delta, -max_dof_delta, max_dof_delta)

			self.cur_targets[:, self.actuated_dof_indices] = self.prev_targets[:, self.actuated_dof_indices] + delta


		else:

			self.cur_targets[:, self.actuated_dof_indices] = scale(actions_delayed,
																   self.hand_dof_lower_limits[self.actuated_dof_indices],
																   self.hand_dof_upper_limits[self.actuated_dof_indices])

		self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,self.actuated_dof_indices] + \
															(1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]


		self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
																		  self.hand_dof_lower_limits[self.actuated_dof_indices], self.hand_dof_upper_limits[self.actuated_dof_indices])

		self.dof_delta = self.cur_targets[:, self.actuated_dof_indices] - self.prev_targets[:, self.actuated_dof_indices]

		self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

		self.prev_actions[:] = self.actions.clone()

	def compute_observations(self):
		AllegroHandDextremeADR.compute_observations(self)

		self.obs_dict["last_actions"][:] = self.base_controller_actions
		self.obs_dict["last_actions_full"][:] = self.actions_full

	def compute_reward(self, actions):
		AllegroHandDextremeADR.compute_reward(self, actions)

		if self.print_success_stat:
			if self.frame % 100 == 0:
				dof_pos_noise_avg = self.dof_pos_noise_scaled.mean(dim=1)
				object_pose_noise_avg = self.cube_pose_noise_scaled.mean(dim=1)
				action_noise_avg = self.action_noise_scaled.mean(dim=1)
				base_action_avg = self.base_controller_actions.mean(dim=1)

				for i in range(16):
					# self.eval_summaries.add_scalar(f"base_actions_avg/actuator_{i+1}", base_action_avg[i].item(), self.frame)
					self.eval_summaries.add_scalar(f"action_noise_avg/actuator_{i+1}", action_noise_avg[i].item(), self.frame)
					self.eval_summaries.add_scalar(f"dof_pos_noise_avg/dof_{i+1}", dof_pos_noise_avg[i].item(), self.frame)

				for i in range(7):
					self.eval_summaries.add_scalar(f"object_pos_noise_avg/value_{i+1}", object_pose_noise_avg[i].item(), self.frame)