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
from typing import Tuple, List 

import itertools
from itertools import permutations
from tkinter import W
from typing import Tuple, Dict, List, Set

import numpy as np

import torch

from isaacgymenvs.tasks.dextreme.allegro_hand_dextreme import AllegroHandDextremeADR

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
		if debug: print("[TASK-AdvActions][DEBUG] Obs shape", np_obs.shape, type(np_obs))
		for i,e in enumerate(self.states):
			if debug: print(f"[TASK-AdvActions][DEBUG] State {i} shape", e.shape, type(e))
		input_dict = {
			'obs' : np_obs,
			'out_state.1' : self.states[0],
			'hidden_state.1' : self.states[1],
		}
		if debug: print("[TASK-AdvActions][DEBUG] Running ONNX model")
		mu, out_states, hidden_states = self._model.run(None, input_dict)
		if debug: print("[TASK-AdvActions][DEBUG] ONNX model ran successfully")
		if debug: print(f"[TASK-AdvActions][DEBUG] mu shape:", mu.shape)
		if debug: print(f"[TASK-AdvActions][DEBUG] out_states shape:", out_states.shape)
		if debug: print(f"[TASK-AdvActions][DEBUG] hidden_states shape:", hidden_states.shape)
		self.states = (out_states, hidden_states)
		current_action = torch.tensor(mu).to(self.device)
		
		# TODO: Change hardcoded action high and low
		return self.rescale_actions(-1.0, 1.0, torch.clamp(current_action, -1.0, 1.0))

class AllegroHandDextremeAdversarialActions(AllegroHandDextremeADR):
	# def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
	# 	super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
	# 	if self.print_success_stat:
	# 		self.last_action_noise = torch.zeros((16, self.num_envs), dtype=torch.float, device=self.device)				
	# 		self.last_base_action = torch.zeros((16, self.num_envs), dtype=torch.float, device=self.device)				

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
					self.eval_summaries.add_scalar(f"base_actions_avg/actuator_{i+1}", last_base_action_avg[i].item(), self.frame)
					self.eval_summaries.add_scalar(f"action_noises_avg/actuator_{i+1}", last_action_noise_avg[i].item(), self.frame)
