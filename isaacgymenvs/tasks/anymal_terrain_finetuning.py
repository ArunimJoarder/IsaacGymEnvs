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
from isaacgymenvs.tasks.anymal_terrain import AnymalTerrain
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
			print("[TASK-AnymalTerrainFinetuning][DEBUG] Obs shape", np_obs.shape, type(np_obs))

		input_dict = {
			'obs' : np_obs,
		}
		
		if debug: print("[TASK-AnymalTerrainFinetuning][DEBUG] Running ONNX model")
		
		mu = self._model.run(None, input_dict)
		if debug: 
			print("[TASK-AnymalTerrainFinetuning][DEBUG] ONNX model ran successfully")
			print(f"[TASK-AnymalTerrainFinetuning][DEBUG] mu shape:", mu.shape)

		current_action = torch.tensor(mu[0]).to(self.device)
		
		# TODO: Change hardcoded action high and low
		return self.rescale_actions(-1.0, 1.0, torch.clamp(current_action, -1.0, 1.0))

class AnymalTerrainFinetuningResidualActions(AnymalTerrain, VecTask):
	def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
		AnymalTerrain.__init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

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
		AnymalTerrain.pre_physics_step(self, actions)

	def compute_observations(self):
		AnymalTerrain.compute_observations(self)
		self.base_obs_buf = self.obs_buf
		self.base_obs_dict["obs"] = torch.clamp(self.base_obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

		self.obs_buf = torch.cat([self.obs_buf, self.base_actions], dim=-1)