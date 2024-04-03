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
from isaacgymenvs.tasks.anymal import Anymal, compute_anymal_observations
from isaacgymenvs.tasks.base.vec_task import VecTask

import onnxruntime as ort

debug = False

class BaseControllerPlugin:
	def __init__(self, onnx_model_checkpoint, device) -> None:
		sess_options = ort.SessionOptions()
		sess_options.inter_op_num_threads = 8
		sess_options.intra_op_num_threads = 8
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
			print("[TASK-AnymalFinetuning][DEBUG] Obs shape", np_obs.shape, type(np_obs))

		input_dict = {
			'obs' : np_obs,
		}
		
		if debug: print("[TASK-AnymalFinetuning][DEBUG] Running ONNX model")
		
		mu = self._model.run(None, input_dict)
		if debug: 
			print("[TASK-AnymalFinetuning][DEBUG] ONNX model ran successfully")
			print(f"[TASK-AnymalFinetuning][DEBUG] mu shape:", mu.shape)

		current_action = torch.tensor(mu[0]).to(self.device)
		
		# TODO: Change hardcoded action high and low
		return self.rescale_actions(-1.0, 1.0, torch.clamp(current_action, -1.0, 1.0))

class AnymalFinetuningResidualActions(Anymal, VecTask):
	def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

		self.cfg = cfg
		
		# normalization
		self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
		self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
		self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
		self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
		self.action_scale = self.cfg["env"]["control"]["actionScale"]

		# reward scales
		self.rew_scales = {}
		self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
		self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
		self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]

		# randomization
		self.randomization_params = self.cfg["task"]["randomization_params"]
		self.randomize = self.cfg["task"]["randomize"]

		# command ranges
		self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
		self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
		self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

		# plane params
		self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
		self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
		self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

		# base init state
		pos = self.cfg["env"]["baseInitState"]["pos"]
		rot = self.cfg["env"]["baseInitState"]["rot"]
		v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
		v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
		state = pos + rot + v_lin + v_ang

		self.base_init_state = state

		# default joint positions
		self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

		self.cfg["env"]["numObservations"] = 48 + 12
		self.cfg["env"]["numActions"] = 12

		VecTask.__init__(self, config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

		# other
		self.dt = self.sim_params.dt
		self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
		self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
		self.Kp = self.cfg["env"]["control"]["stiffness"]
		self.Kd = self.cfg["env"]["control"]["damping"]

		for key in self.rew_scales.keys():
			self.rew_scales[key] *= self.dt

		if self.viewer != None:
			p = self.cfg["env"]["viewer"]["pos"]
			lookat = self.cfg["env"]["viewer"]["lookat"]
			cam_pos = gymapi.Vec3(p[0], p[1], p[2])
			cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
			self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

		# get gym state tensors
		actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
		dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
		net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
		torques = self.gym.acquire_dof_force_tensor(self.sim)

		self.gym.refresh_dof_state_tensor(self.sim)
		self.gym.refresh_actor_root_state_tensor(self.sim)
		self.gym.refresh_net_contact_force_tensor(self.sim)
		self.gym.refresh_dof_force_tensor(self.sim)

		# create some wrapper tensors for different slices
		self.root_states = gymtorch.wrap_tensor(actor_root_state)
		self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
		self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
		self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
		self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
		self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)

		self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
		self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
		self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
		self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]
		self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)

		for i in range(self.cfg["env"]["numActions"]):
			name = self.dof_names[i]
			angle = self.named_default_joint_angles[name]
			self.default_dof_pos[:, i] = angle

		# initialize some data used later on
		self.extras = {}
		self.initial_root_states = self.root_states.clone()
		self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
		self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
		self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

		self.print_eval_stats = self.cfg["env"]["printEvalStats"]
		self.init_summary_writer()

		self.reset_idx(torch.arange(self.num_envs, device=self.device))


		self.base_obs_buf = torch.zeros((self.num_envs, self.num_obs - self.num_actions), device=self.device, dtype=torch.float)
		self.base_obs_dict = {}
		self.base_obs_dict["obs"] = torch.clamp(self.base_obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
		self.base_controller_checkpoint = self.cfg["onnx_model_checkpoint"]
		self.base_controller = BaseControllerPlugin(self.base_controller_checkpoint, self.device)

	def pre_physics_step(self, delta_actions):
		self.delta_actions = delta_actions
		self.base_actions = self.base_controller.get_action(self.base_obs_dict)
		actions = self.base_actions + self.delta_actions
		actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
		Anymal.pre_physics_step(self, actions)

	def compute_observations(self):
		self.gym.refresh_dof_state_tensor(self.sim)  # done in step
		self.gym.refresh_actor_root_state_tensor(self.sim)
		self.gym.refresh_net_contact_force_tensor(self.sim)
		self.gym.refresh_dof_force_tensor(self.sim)

		self.base_obs_buf[:] = compute_anymal_observations(  # tensors
														self.root_states,
														self.commands,
														self.dof_pos,
														self.default_dof_pos,
														self.dof_vel,
														self.gravity_vec,
														self.actions,
														# scales
														self.lin_vel_scale,
														self.ang_vel_scale,
														self.dof_pos_scale,
														self.dof_vel_scale
		)
		self.base_obs_dict["obs"] = torch.clamp(self.base_obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

		self.obs_buf = torch.cat([self.base_obs_buf, self.base_actions], dim=-1)