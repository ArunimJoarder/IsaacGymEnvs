# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
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

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.gymtorch import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.ant import Ant, compute_ant_observations
from isaacgymenvs.tasks.base.vec_task import VecTask

import onnxruntime as ort

debug = False

class BaseControllerPlugin:
	def __init__(self, onnx_model_checkpoint, device) -> None:
		sess_options = ort.SessionOptions()
		# sess_options.inter_op_num_threads = 8
		# sess_options.intra_op_num_threads = 8
		# sess_options.log_severity_level = 0
		self._model = ort.InferenceSession(onnx_model_checkpoint, sess_options=sess_options, providers=["CUDAExecutionProvider"])
		self.device = device

	def obs_to_numpy(self, obs):
		obs = obs["obs"].cpu().numpy().astype(np.float32)
		return obs
		
	def rescale_actions(self, low, high, action):
		d = (high - low) / 2.0
		m = (high + low) / 2.0
		scaled_action =  action * d + m
		return scaled_action

	def get_action(self, obs):
		np_obs = self.obs_to_numpy(obs)
		
		if debug: 
			print("[TASK-AntFinetuning][DEBUG] Obs shape", np_obs.shape, type(np_obs))

		input_dict = {
			'obs' : np_obs,
		}
		
		if debug: print("[TASK-AntFinetuning][DEBUG] Running ONNX model")
		
		mu = self._model.run(None, input_dict)
		if debug: 
			print("[TASK-AntFinetuning][DEBUG] ONNX model ran successfully")
			print(f"[TASK-AntFinetuning][DEBUG] mu shape:", mu.shape)

		current_action = torch.tensor(mu[0]).to(self.device)
		
		# TODO: Change hardcoded action high and low
		return self.rescale_actions(-1.0, 1.0, torch.clamp(current_action, -1.0, 1.0))

class AntFinetuningResidualActions(Ant, VecTask):
	def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
		self.cfg = cfg

		self.max_episode_length = self.cfg["env"]["episodeLength"]

		self.randomization_params = self.cfg["task"]["randomization_params"]
		self.randomize = self.cfg["task"]["randomize"]
		self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
		self.contact_force_scale = self.cfg["env"]["contactForceScale"]
		self.power_scale = self.cfg["env"]["powerScale"]
		self.heading_weight = self.cfg["env"]["headingWeight"]
		self.up_weight = self.cfg["env"]["upWeight"]
		self.actions_cost_scale = self.cfg["env"]["actionsCost"]
		self.energy_cost_scale = self.cfg["env"]["energyCost"]
		self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
		self.death_cost = self.cfg["env"]["deathCost"]
		self.termination_height = self.cfg["env"]["terminationHeight"]

		self.debug_viz = self.cfg["env"]["enableDebugVis"]
		self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
		self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
		self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

		self.cfg["env"]["numObservations"] = 60 + 8
		self.cfg["env"]["numActions"] = 8

		VecTask.__init__(self, config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

		if self.viewer != None:
			cam_pos = gymapi.Vec3(50.0, 25.0, 2.4)
			cam_target = gymapi.Vec3(45.0, 25.0, 0.0)
			self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

		# get gym GPU state tensors
		actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
		dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
		sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

		sensors_per_env = 4
		self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

		self.gym.refresh_dof_state_tensor(self.sim)
		self.gym.refresh_actor_root_state_tensor(self.sim)

		self.root_states = gymtorch.wrap_tensor(actor_root_state)
		self.initial_root_states = self.root_states.clone()
		self.initial_root_states[:, 7:13] = 0  # set lin_vel and ang_vel to 0

		# create some wrapper tensors for different slices
		self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
		self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
		self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
		self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
		zero_tensor = torch.tensor([0.0], device=self.device)
		self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor,
									 	   self.dof_limits_lower,
										   torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper, self.initial_dof_pos))
		self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device, dtype=torch.float)

		# initialize some data used later on
		self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
		self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
		self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

		self.basis_vec0 = self.heading_vec.clone()
		self.basis_vec1 = self.up_vec.clone()

		self.targets = to_torch([1000, 0, 0], device=self.device).repeat((self.num_envs, 1))
		self.target_dirs = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
		self.dt = self.cfg["sim"]["dt"]
		self.potentials = to_torch([-1000./self.dt], device=self.device).repeat(self.num_envs)
		self.prev_potentials = self.potentials.clone()

		self.base_obs_buf = torch.zeros((self.num_envs, 60), device=self.device, dtype=torch.float)
		self.base_obs_dict = {}
		self.base_obs_dict["obs"] = torch.clamp(self.base_obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
		self.base_controller_checkpoint = self.cfg["onnx_model_checkpoint"]
		self.base_controller = BaseControllerPlugin(self.base_controller_checkpoint, self.device)

	def pre_physics_step(self, delta_actions):
		self.delta_actions = delta_actions
		self.base_actions = self.base_controller.get_action(self.base_obs_dict)
		actions = self.base_actions + 0 * self.delta_actions
		actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
		
		Ant.pre_physics_step(self, actions)

	def compute_observations(self):
		self.gym.refresh_dof_state_tensor(self.sim)
		self.gym.refresh_actor_root_state_tensor(self.sim)
		self.gym.refresh_force_sensor_tensor(self.sim)

		self.base_obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:] = compute_ant_observations(
			self.obs_buf, self.root_states, self.targets, self.potentials,
			self.inv_start_rot, self.dof_pos, self.dof_vel,
			self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
			self.vec_sensor_tensor, self.actions, self.dt, self.contact_force_scale,
			self.basis_vec0, self.basis_vec1, self.up_axis_idx)
		
		self.obs_buf[:] = torch.hstack([self.base_obs_buf, self.base_actions])
		self.base_obs_dict["obs"] = torch.clamp(self.base_obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

