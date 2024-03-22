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
from isaacgymenvs.tasks.franka_cube_stack import FrankaCubeStack
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
			print("[TASK-FrankaCubeStackFinetuning][DEBUG] Obs shape", np_obs.shape, type(np_obs))

		input_dict = {
			'obs' : np_obs,
		}
		
		if debug: print("[TASK-FrankaCubeStackFinetuning][DEBUG] Running ONNX model")
		
		mu = self._model.run(None, input_dict)
		if debug: 
			print("[TASK-FrankaCubeStackFinetuning][DEBUG] ONNX model ran successfully")
			print(f"[TASK-FrankaCubeStackFinetuning][DEBUG] mu shape:", mu.shape)

		current_action = torch.tensor(mu[0]).to(self.device)
		
		# TODO: Change hardcoded action high and low
		return self.rescale_actions(-1.0, 1.0, torch.clamp(current_action, -1.0, 1.0))

class FrankaCubeStackFinetuningResidualActions(FrankaCubeStack, VecTask):
	def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
		self.cfg = cfg

		self.max_episode_length = self.cfg["env"]["episodeLength"]

		self.action_scale = self.cfg["env"]["actionScale"]
		self.start_position_noise = self.cfg["env"]["startPositionNoise"]
		self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
		self.franka_position_noise = self.cfg["env"]["frankaPositionNoise"]
		self.franka_rotation_noise = self.cfg["env"]["frankaRotationNoise"]
		self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
		self.aggregate_mode = self.cfg["env"]["aggregateMode"]

		# Create dicts to pass to reward function
		self.reward_settings = {
			"r_dist_scale": self.cfg["env"]["distRewardScale"],
			"r_lift_scale": self.cfg["env"]["liftRewardScale"],
			"r_align_scale": self.cfg["env"]["alignRewardScale"],
			"r_stack_scale": self.cfg["env"]["stackRewardScale"],
		}

		# Controller type
		self.control_type = self.cfg["env"]["controlType"]
		assert self.control_type in {"osc", "joint_tor"},\
			"Invalid control type specified. Must be one of: {osc, joint_tor}"

		# dimensions
		# obs include: cubeA_pose (7) + cubeB_pos (3) + eef_pose (7) + q_gripper (2)
		self.cfg["env"]["numObservations"] = 26 if self.control_type == "osc" else 34
		# actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
		self.cfg["env"]["numActions"] = 7 if self.control_type == "osc" else 8

		# Values to be filled in at runtime
		self.states = {}                        # will be dict filled with relevant states to use for reward calculation
		self.handles = {}                       # will be dict mapping names to relevant sim handles
		self.num_dofs = None                    # Total number of DOFs per env
		self.actions = None                     # Current actions to be deployed
		self._init_cubeA_state = None           # Initial state of cubeA for the current env
		self._init_cubeB_state = None           # Initial state of cubeB for the current env
		self._cubeA_state = None                # Current state of cubeA for the current env
		self._cubeB_state = None                # Current state of cubeB for the current env
		self._cubeA_id = None                   # Actor ID corresponding to cubeA for a given env
		self._cubeB_id = None                   # Actor ID corresponding to cubeB for a given env

		# Tensor placeholders
		self._root_state = None             # State of root body        (n_envs, 13)
		self._dof_state = None  # State of all joints       (n_envs, n_dof)
		self._q = None  # Joint positions           (n_envs, n_dof)
		self._qd = None                     # Joint velocities          (n_envs, n_dof)
		self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
		self._contact_forces = None     # Contact forces in sim
		self._eef_state = None  # end effector state (at grasping point)
		self._eef_lf_state = None  # end effector state (at left fingertip)
		self._eef_rf_state = None  # end effector state (at left fingertip)
		self._j_eef = None  # Jacobian for end effector
		self._mm = None  # Mass matrix
		self._arm_control = None  # Tensor buffer for controlling arm
		self._gripper_control = None  # Tensor buffer for controlling gripper
		self._pos_control = None            # Position actions
		self._effort_control = None         # Torque actions
		self._franka_effort_limits = None        # Actuator effort limits for franka
		self._global_indices = None         # Unique indices corresponding to all envs in flattened array

		self.debug_viz = self.cfg["env"]["enableDebugVis"]

		self.up_axis = "z"
		self.up_axis_idx = 2

		VecTask.__init__(self, config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

		self.extras["custom"] = {}
		self.stack_buf = torch.zeros(
			self.num_envs, device=self.device, dtype=torch.bool)
		
		# Franka defaults
		self.franka_default_dof_pos = to_torch(
			[0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035], device=self.device
		)

		# OSC Gains
		self.kp = to_torch([150.] * 6, device=self.device)
		self.kd = 2 * torch.sqrt(self.kp)
		self.kp_null = to_torch([10.] * 7, device=self.device)
		self.kd_null = 2 * torch.sqrt(self.kp_null)
		#self.cmd_limit = None                   # filled in later

		# Set control limits
		self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
		self.control_type == "osc" else self._franka_effort_limits[:7].unsqueeze(0)

		# Reset all environments
		self.reset_idx(torch.arange(self.num_envs, device=self.device))

		# Refresh tensors
		self._refresh()

		self.num_base_obs = 19 if self.control_type == "osc" else 26

		self.base_obs_buf = torch.zeros((self.num_envs, self.num_base_obs), device=self.device, dtype=torch.float)
		self.base_obs_dict = {}
		self.base_obs_dict["obs"] = torch.clamp(self.base_obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
		self.base_controller_checkpoint = self.cfg["onnx_model_checkpoint"]
		self.base_controller = BaseControllerPlugin(self.base_controller_checkpoint, self.device)

	def pre_physics_step(self, delta_actions):
		self.delta_actions = delta_actions
		self.base_actions = self.base_controller.get_action(self.base_obs_dict)
		actions = self.base_actions + self.delta_actions
		actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
		FrankaCubeStack.pre_physics_step(self, actions)

	def compute_observations(self):
		self.base_obs_buf = FrankaCubeStack.compute_observations(self)
		self.base_obs_dict["obs"] = torch.clamp(self.base_obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

		self.obs_buf = torch.cat([self.obs_buf, self.base_actions], dim=-1)