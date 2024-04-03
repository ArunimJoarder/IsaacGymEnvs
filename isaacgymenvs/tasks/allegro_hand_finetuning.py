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
from isaacgymenvs.tasks.allegro_hand import AllegroHand
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
			print("[TASK-AllegroHandFinetuning][DEBUG] Obs shape", np_obs.shape, type(np_obs))

		input_dict = {
			'obs' : np_obs,
		}
		
		if debug: print("[TASK-AllegroHandFinetuning][DEBUG] Running ONNX model")
		
		mu = self._model.run(None, input_dict)
		if debug: 
			print("[TASK-AllegroHandFinetuning][DEBUG] ONNX model ran successfully")
			print(f"[TASK-AllegroHandFinetuning][DEBUG] mu shape:", mu.shape)

		current_action = torch.tensor(mu[0]).to(self.device)
		
		# TODO: Change hardcoded action high and low
		return self.rescale_actions(-1.0, 1.0, torch.clamp(current_action, -1.0, 1.0))

class AllegroHandFinetuningResidualActions(AllegroHand, VecTask):
	def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

		self.cfg = cfg

		self.aggregate_mode = self.cfg["env"]["aggregateMode"]

		self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
		self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
		self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
		self.success_tolerance = self.cfg["env"]["successTolerance"]
		self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
		self.fall_dist = self.cfg["env"]["fallDistance"]
		self.fall_penalty = self.cfg["env"]["fallPenalty"]
		self.rot_eps = self.cfg["env"]["rotEps"]

		self.vel_obs_scale = 0.2  # scale factor of velocity based observations
		self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

		self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
		self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
		self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
		self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

		self.force_scale = self.cfg["env"].get("forceScale", 0.0)
		self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
		self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
		self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

		self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
		self.use_relative_control = self.cfg["env"]["useRelativeControl"]
		self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

		self.debug_viz = self.cfg["env"]["enableDebugVis"]

		self.max_episode_length = self.cfg["env"]["episodeLength"]
		self.reset_time = self.cfg["env"].get("resetTime", -1.0)
		self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
		self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
		self.av_factor = self.cfg["env"].get("averFactor", 0.1)

		self.object_type = self.cfg["env"]["objectType"]
		assert self.object_type in ["block", "egg", "pen"]

		self.ignore_z = (self.object_type == "pen")

		self.asset_files_dict = {
			"block": "urdf/objects/cube_multicolor.urdf",
			"egg": "mjcf/open_ai_assets/hand/egg.xml",
			"pen": "mjcf/open_ai_assets/hand/pen.xml"
		}

		if "asset" in self.cfg["env"]:
			self.asset_files_dict["block"] = self.cfg["env"]["asset"].get("assetFileNameBlock", self.asset_files_dict["block"])
			self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get("assetFileNameEgg", self.asset_files_dict["egg"])
			self.asset_files_dict["pen"] = self.cfg["env"]["asset"].get("assetFileNamePen", self.asset_files_dict["pen"])

		# can be "full_no_vel", "full", "full_state"
		self.obs_type = self.cfg["env"]["observationType"]

		if not (self.obs_type in ["full_no_vel", "full", "full_state"]):
			raise Exception(
				"Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]")

		print("Obs type:", self.obs_type)

		self.num_obs_dict = {
			"full_no_vel": 50 + 16,
			"full": 72 + 16,
			"full_state": 88 + 16
		}

		self.up_axis = 'z'

		self.use_vel_obs = False
		self.fingertip_obs = True
		self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

		num_states = 0
		if self.asymmetric_obs:
			num_states = 88

		self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
		self.cfg["env"]["numStates"] = num_states
		self.cfg["env"]["numActions"] = 16

		VecTask.__init__(self, config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

		self.dt = self.sim_params.dt
		control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
		if self.reset_time > 0.0:
			self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.dt)))
			print("Reset time: ", self.reset_time)
			print("New episode length: ", self.max_episode_length)

		if self.viewer != None:
			cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
			cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
			self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

		# get gym GPU state tensors
		actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
		dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
		rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

		if self.obs_type == "full_state" or self.asymmetric_obs:
		#     sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
		#     self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

				dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
				self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_shadow_hand_dofs)

		self.gym.refresh_actor_root_state_tensor(self.sim)
		self.gym.refresh_dof_state_tensor(self.sim)
		self.gym.refresh_rigid_body_state_tensor(self.sim)

		# create some wrapper tensors for different slices
		self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
		self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
		self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
		self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
		self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]

		self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
		self.num_bodies = self.rigid_body_states.shape[1]

		self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

		self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
		print("Num dofs: ", self.num_dofs)

		self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
		self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

		self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
		self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
		self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
		self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

		self.reset_goal_buf = self.reset_buf.clone()
		self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
		self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

		self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

		self.total_successes = 0
		self.total_resets = 0

		# object apply random forces parameters
		self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
		self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
		self.random_force_prob = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
											* torch.rand(self.num_envs, device=self.device) + torch.log(self.force_prob_range[1]))

		self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

		self.base_obs_buf = torch.zeros((self.num_envs, self.num_obs - self.num_actions), device=self.device, dtype=torch.float)
		self.base_obs_dict = {}
		self.base_obs_dict["obs"] = torch.clamp(self.base_obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
		self.base_controller_checkpoint = self.cfg["onnx_model_checkpoint"]
		self.base_controller = BaseControllerPlugin(self.base_controller_checkpoint, self.device)
        
		self.print_eval_stats = self.cfg["env"]["printEvalStats"]
		self.init_summary_writer()

	def pre_physics_step(self, delta_actions):
		self.delta_actions = delta_actions
		self.base_actions = self.base_controller.get_action(self.base_obs_dict)
		actions = self.base_actions + self.delta_actions
		actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
		AllegroHand.pre_physics_step(self, actions)

	def compute_observations(self):
		AllegroHand.compute_observations(self)
		self.base_obs_buf = self.obs_buf[:, :self.num_obs - self.num_actions]
		self.base_obs_dict["obs"] = torch.clamp(self.base_obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

		self.obs_buf[:, -self.num_actions:] = self.base_actions