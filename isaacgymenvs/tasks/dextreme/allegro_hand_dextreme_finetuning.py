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
from isaacgymenvs.tasks.dextreme.allegro_hand_dextreme_adversarial import BaseControllerPlugin
from isaacgymenvs.utils.torch_jit_utils import scale, unscale, tensor_clamp  
from isaacgymenvs.utils.torch_jit_utils import quat_from_euler_xyz, quat_mul, quat_conjugate

from gym import spaces
from isaacgym import gymtorch
import onnxruntime as ort

import matplotlib.pyplot as plt

debug = False

class BaseNoiseGeneratorPlugin:
	def __init__(self, onnx_model_checkpoint, device) -> None:
		sess_options = ort.SessionOptions()
		# sess_options.inter_op_num_threads = 4
		# sess_options.intra_op_num_threads = 4
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

		self.delta_actions = delta_actions

		actions = self.base_actions + delta_actions
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

	def compute_reward(self, actions):
		return super().compute_reward(self.delta_actions)
