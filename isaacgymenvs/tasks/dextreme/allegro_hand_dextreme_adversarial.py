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
from isaacgymenvs.tasks.dextreme.allegro_hand_dextreme import AllegroHandDextremeADR, AllegroHandDextreme, AllegroHandDextremeManualDR
from isaacgymenvs.utils.torch_jit_utils import scale, unscale, tensor_clamp  
from isaacgymenvs.utils.torch_jit_utils import quat_from_euler_xyz, quat_mul, quat_conjugate

from gym import spaces
from isaacgym import gymtorch
import onnxruntime as ort

import matplotlib.pyplot as plt

debug = False

class BaseControllerPlugin:
	def __init__(self, onnx_model_checkpoint, device) -> None:
		sess_options = ort.SessionOptions()
		# sess_options.inter_op_num_threads = 8
		# sess_options.intra_op_num_threads = 8
		# sess_options.log_severity_level = 0
		self._model = ort.InferenceSession(onnx_model_checkpoint, sess_options=sess_options, providers=["CUDAExecutionProvider"])
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

class AllegroHandDextremeAdversarialObservationsAndActions(AllegroHandDextremeADR, AllegroHandDextremeManualDR, AllegroHandDextreme, ADRVecTask):
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
		adr_cfg = self.cfg["task"].get("adr", {})
		self.use_adr = adr_cfg.get("use_adr", False)
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

		self.cfg["env"]["numActions"] = 16 + 3 + 3 + 16

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

		self.realtime_plots = self.cfg["plot_noises"]
		if self.realtime_plots:
			if self.num_envs <= 4:
				self.rt_plot_fig_actions, self.rt_plot_ax_actions = plt.subplots(4,4, sharey=True, sharex=True)
				self.rt_plot_fig_obs_dof, self.rt_plot_ax_obs_dof = plt.subplots(4,4, sharey=True, sharex=True)
				self.rt_plot_fig_obs_cube_pose, self.rt_plot_ax_obs_cube_pose = plt.subplots(2,3, sharex=True)

				self.rt_plot_fig_actions.suptitle("Adversarial Action Noises")
				for i, ax in enumerate(self.rt_plot_ax_actions.flat):
					ax.set(xlabel='frame')
					ax.set_title(f"dof_{i+1} Action Noise")

				self.rt_plot_fig_obs_dof.suptitle("Adversarial Observation DOF Noises")
				for i, ax in enumerate(self.rt_plot_ax_obs_dof.flat):
					ax.set(xlabel='frame')
					ax.set_title(f"dof_{i+1} Observation Noise")

				self.rt_plot_fig_obs_cube_pose.suptitle("Adversarial Observation Cube Pose Noises")
				self.rt_plot_ax_obs_cube_pose[0,0].set_title("Cube Pos X Noise")					
				self.rt_plot_ax_obs_cube_pose[0,1].set_title("Cube Pos Y Noise")					
				self.rt_plot_ax_obs_cube_pose[0,2].set_title("Cube Pos Z Noise")					
				self.rt_plot_ax_obs_cube_pose[1,0].set_title("Cube Rot X Noise")					
				self.rt_plot_ax_obs_cube_pose[1,1].set_title("Cube Rot Y Noise")					
				self.rt_plot_ax_obs_cube_pose[1,2].set_title("Cube Rot Z Noise")					
				for ax in self.rt_plot_ax_obs_cube_pose.flat:
					ax.set(xlabel='frame')

				self.rt_plot_fig_obs_cube_pose.show()
				self.rt_plot_fig_obs_dof.show()
				self.rt_plot_fig_actions.show()
				plt.pause(0.0001)

				self.rt_plot_actions_buffer = np.zeros((1, self.num_envs, 16))
				self.rt_plot_obs_dof_buffer = np.zeros((1, self.num_envs, 16))
				self.rt_plot_obs_cube_pos_buffer = np.zeros((1, self.num_envs, 3))
				self.rt_plot_obs_cube_rot_buffer = np.zeros((1, self.num_envs, 3))
				self.rt_plot_frame_buffer = np.zeros(1)

			else:
				self.realtime_plots = False
				print("Real-time plots cannot be plotted with more than 4 agents being simulated at once.") 

	def get_num_obs_dict(self, num_dofs):
		if self.use_adr:
			num_obs_dict = AllegroHandDextremeADR.get_num_obs_dict(self, num_dofs)
		else:
			num_obs_dict = AllegroHandDextremeManualDR.get_num_obs_dict(self, num_dofs)

		num_obs_dict["last_actions_full"] = 16 + 3 + 3 + 16
		num_obs_dict["last_actions"] = 16
		return num_obs_dict

	def _read_cfg(self):
		AllegroHandDextreme._read_cfg(self)
		if self.use_adr:
			self.vel_obs_scale = 1.0  # scale factor of velocity based observations
			self.force_torque_obs_scale = 1.0  # scale factor of velocity based observations
		
		self.base_controller_checkpoint = self.cfg["onnx_model_checkpoint"]
		self.action_noise_penalty_scale = self.cfg["env"]["actionNoisePenaltyScale"]
		self.cube_rot_noise_penalty_scale = self.cfg["env"]["cubeRotNoisePenaltyScale"]
		self.cube_pos_noise_penalty_scale = self.cfg["env"]["cubePosNoisePenaltyScale"]
		self.dof_pos_noise_penalty_scale = self.cfg["env"]["dofPosNoisePenaltyScale"]

		self.test = self.cfg["test"]

		if not self.test:
			self.clip_action_noise = self.cfg["env"].get("clipActionNoiseInit", 0.1)
			self.clip_dof_pos_noise = self.cfg["env"].get("clipDofPosNoiseInit", 0.1)

			self.clip_action_noise_init = self.cfg["env"].get("clipActionNoiseInit", 0.1)
			self.clip_dof_pos_noise_init = self.cfg["env"].get("clipDofPosNoiseInit", 0.1)

			self.clip_action_noise_end = min(self.clip_action_noise_init, self.cfg["env"].get("clipActionNoiseEnd", 0.1))
			self.clip_dof_pos_noise_end = min(self.clip_dof_pos_noise_init, self.cfg["env"].get("clipDofPosNoiseEnd", 0.1))

			self.curriculum_max_steps = self.cfg["env"]["curriculumMaxSteps"]
			self.curriculum_sched_freq = self.cfg["env"]["curriculumScheduleFreq"]
		else:
			self.clip_action_noise = self.cfg["env"].get("clipActionNoiseEnd", 0.1)
			self.clip_dof_pos_noise = self.cfg["env"].get("clipDofPosNoiseEnd", 0.1)	

		self.clip_cube_rot_noise = self.cfg["env"].get("clipCubeRotNoise", 0.1)		
		self.clip_cube_pos_noise = self.cfg["env"].get("clipCubePosNoise", 0.05)		


	def _init_pre_sim_buffers(self):
		AllegroHandDextreme._init_pre_sim_buffers(self)
		if self.use_adr:
			self.cube_pose_refresh_rate = torch.zeros(self.cfg["env"]["numEnvs"], device=self.sim_device, dtype=torch.long)
			# offset so not all the environments have it each time
			self.cube_pose_refresh_offset = torch.zeros(self.cfg["env"]["numEnvs"], device=self.sim_device, dtype=torch.long)
			
			# stores previous actions
			self.prev_actions_queue = torch.zeros(self.cfg["env"]["numEnvs"], self.action_latency_max + 1, self.cfg["env"]["numActions"], dtype=torch.float, device=self.sim_device)
			
			# tensors to store random affine transforms
			self.affine_actions_scaling = torch.ones(self.cfg["env"]["numEnvs"], 16, dtype=torch.float, device=self.sim_device)
			self.affine_actions_additive = torch.zeros(self.cfg["env"]["numEnvs"], 16, dtype=torch.float, device=self.sim_device)

			self.affine_cube_pose_scaling = torch.ones(self.cfg["env"]["numEnvs"], 7, dtype=torch.float, device=self.sim_device)
			self.affine_cube_pose_additive = torch.zeros(self.cfg["env"]["numEnvs"], 7, dtype=torch.float, device=self.sim_device)

			self.affine_dof_pos_scaling = torch.ones(self.cfg["env"]["numEnvs"], 16, dtype=torch.float, device=self.sim_device)
			self.affine_dof_pos_additive = torch.zeros(self.cfg["env"]["numEnvs"], 16, dtype=torch.float, device=self.sim_device)

			self.action_latency = torch.zeros(self.cfg["env"]["numEnvs"], dtype=torch.long, device=self.sim_device)

	def _init_post_sim_buffers(self):
		AllegroHandDextreme._init_post_sim_buffers(self)
		self.prev_actions = torch.zeros(self.num_envs, 16, dtype=torch.float, device=self.device)
		self.prev_actions_queue = torch.zeros(self.cfg["env"]["numEnvs"], self.action_latency_max + 1, 16, dtype=torch.float, device=self.sim_device)

		if not self.use_adr:
			# We could potentially update this regularly
			self.action_delay_prob = self.action_delay_prob_max * \
				torch.rand(self.cfg["env"]["numEnvs"], dtype=torch.float, device=self.device)
			
			# inverse refresh rate for each environment
			self.cube_pose_refresh_rate = torch.randint(1, self.max_skip_obs+1, size=(self.num_envs,), device=self.device)
			# offset so not all the environments have it each time
			self.cube_pose_refresh_offset = torch.randint(0, self.max_skip_obs, size=(self.num_envs,), device=self.device)

		self.base_controller = BaseControllerPlugin(self.base_controller_checkpoint, self.device)

	def update_curriculum_clips(self):
		if self.last_step > 0 and self.last_step % self.curriculum_sched_freq == 0:

			sched_scaling = 1.0 / self.curriculum_max_steps * min(self.last_step, self.curriculum_max_steps)
			self.clip_action_noise = self.clip_action_noise_init + (self.clip_action_noise_end - self.clip_action_noise_init) * \
										sched_scaling
			self.clip_dof_pos_noise = self.clip_dof_pos_noise_init + (self.clip_dof_pos_noise_end - self.clip_dof_pos_noise_init) * \
										sched_scaling			
			print('clip action noise: {}'.format(self.clip_action_noise))
			print('clip dof pos noise: {}'.format(self.clip_dof_pos_noise))
			print('last_step: {}'.format(self.last_step), ' scheduled steps: {}'.format(self.curriculum_max_steps))

			self.extras['annealing/clip_action_noise'] = self.clip_action_noise
			self.extras['annealing/clip_dof_pos_noise'] = self.clip_dof_pos_noise


	def pre_physics_step(self, obs_and_action_noises):
		# Anneal action moving average 
		self.update_action_moving_average()
	   
		if not self.test:
			self.update_curriculum_clips()

		env_ids_reset = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
		goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

		if self.randomize and not self.use_adr:
			self.apply_randomizations(dr_params=self.randomization_params, randomize_buf=None, randomisation_callback=self.randomisation_callback)

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

		self.cube_pos_noise = obs_and_action_noises[:,16:19]
		self.cube_rot_noise = obs_and_action_noises[:,19:22]
		self.dof_pos_noise  = obs_and_action_noises[:,22:38]
		self.action_noise   = obs_and_action_noises[:,0:16]

		self.cube_pos_noise_scaled = self.cube_pos_noise * self.clip_cube_pos_noise
		self.cube_rot_noise_scaled = self.cube_rot_noise * self.clip_cube_rot_noise
		self.dof_pos_noise_scaled = self.dof_pos_noise * self.clip_dof_pos_noise
		self.action_noise_scaled = self.action_noise * self.clip_action_noise

		self.add_noise_to_observations()
		self.base_controller_actions = self.base_controller.get_action(self.obs_dict)
		actions = torch.clamp(self.base_controller_actions + self.action_noise_scaled, -1.0, 1.0)

		self.actions_full = obs_and_action_noises.clone().to(self.device)

		self.apply_actions(actions)
		self.apply_random_forces()

	def add_noise_to_observations(self):
		if self.use_adr:
			noisy_cube_pos = self.obs_dict["object_pose_cam_randomized"][:, 0:3] + self.cube_pos_noise_scaled

			cube_rot_noise_quat = quat_from_euler_xyz(self.cube_rot_noise_scaled[:, 0], 
													  self.cube_rot_noise_scaled[:, 1],
													  self.cube_rot_noise_scaled[:, 2])
			
			cube_rot = self.obs_dict["object_pose_cam_randomized"][:, 3:7]
			noisy_cube_rot = quat_mul(cube_rot, cube_rot_noise_quat)

			quat_diff = quat_mul(cube_rot, quat_conjugate(noisy_cube_rot))
			rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
			self.object_rot_noise = rot_dist

			noisy_cube_pose_obs = torch.cat((noisy_cube_pos, noisy_cube_rot), axis=-1)
			
			self.obs_dict["object_pose_cam_randomized"] = noisy_cube_pose_obs
			self.obs_dict["dof_pos_randomized"] = self.obs_dict["dof_pos_randomized"] + self.dof_pos_noise_scaled
		else:
			pass

	def apply_action_noise_latency(self):
		if self.use_adr:
			action_delay_mask = (torch.rand(self.num_envs, device=self.device) < self.get_adr_tensor("action_delay_prob")).view(-1, 1)			

			actions = \
					self.prev_actions_queue[torch.arange(self.prev_actions_queue.shape[0]), self.action_latency] * ~action_delay_mask \
					+ self.prev_actions * action_delay_mask
			
			white_noise = self.sample_gaussian_adr("affine_action_white", self.all_env_ids, trailing_dim=16)
			actions = self.affine_actions_scaling * actions + self.affine_actions_additive + white_noise
					
			return actions
		else:
			# anneal action latency 
			if self.randomize:

				self.cur_action_latency = 1.0 / self.action_latency_scheduled_steps \
					* min(self.last_step, self.action_latency_scheduled_steps)

				self.cur_action_latency = min(max(int(self.cur_action_latency), self.action_latency_min), self.action_latency_max)

				self.extras['annealing/cur_action_latency_max'] = self.cur_action_latency

				self.action_latency = torch.randint(0, self.cur_action_latency + 1, \
					size=(self.cfg["env"]["numEnvs"],), dtype=torch.long, device=self.device)

			# probability of not updating the action this step (on top of the delay)
			action_delay_mask = (torch.rand(self.num_envs, device=self.device) > self.action_delay_prob).view(-1, 1)

			actions_delayed = \
				self.prev_actions_queue[torch.arange(self.prev_actions_queue.shape[0]), self.action_latency] * action_delay_mask \
				+ self.prev_actions * ~action_delay_mask
			
			return actions_delayed
		
	def apply_randomizations(self, dr_params, randomize_buf=None, adr_objective=None, randomisation_callback=None):

		AllegroHandDextreme.apply_randomizations(self, dr_params, randomize_buf, adr_objective, randomisation_callback=self.randomisation_callback)
		if self.use_adr:
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

	def get_rna_alpha(self):
		if self.use_adr:
			return AllegroHandDextremeADR.get_rna_alpha(self)
		else:
			return AllegroHandDextremeManualDR.get_rna_alpha(self)

	def compute_observations(self):
		if self.use_adr:
			AllegroHandDextremeADR.compute_observations(self)
		else:
			AllegroHandDextremeManualDR.compute_observations(self)
			
		self.obs_dict["last_actions"][:] = self.base_controller_actions
		self.obs_dict["last_actions_full"][:] = self.actions_full

	def compute_reward(self, actions):

		self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], \
		self.hold_count_buf[:], self.successes[:], self.consecutive_successes[:], \
		dist_rew, rot_rew, action_penalty, action_delta_penalty, velocity_penalty, reach_goal_rew, fall_rew, timeout_rew = compute_hand_reward(
			self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.hold_count_buf, self.cur_targets, self.prev_targets,
			self.dof_vel, self.successes, self.consecutive_successes, self.max_episode_length,
			self.object_pos, self.object_rot, self.goal_pos, self.goal_rot, self.dist_reward_scale, self.rot_reward_scale, self.rot_eps,
			self.actions, self.action_penalty_scale, self.action_delta_penalty_scale,
			self.action_noise, self.action_noise_penalty_scale,
			self.cube_rot_noise, self.cube_rot_noise_penalty_scale,
			self.cube_pos_noise, self.cube_pos_noise_penalty_scale,
			self.dof_pos_noise, self.dof_pos_noise_penalty_scale,
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

				# dof_pos_noise_abs_avg = torch.abs(self.dof_pos_noise_scaled).mean(dim=0)
				# object_rot_noise_abs_avg = torch.abs(self.cube_rot_noise_scaled).mean(dim=0)
				# object_pos_noise_abs_avg = torch.abs(self.cube_pos_noise_scaled).mean(dim=0)
				# action_noise_abs_avg = torch.abs(self.action_noise_scaled).mean(dim=0)

				# # print(self.dof_pos_noise_scaled.size())
				# # print(dof_pos_noise_abs_avg.size())
				# actions_abs_avg = torch.abs(self.actions).mean(dim=0)

				# for i in range(16):
				# 	print(f"action_abs_avg/dof_{i+1} = ", actions_abs_avg[i].item())
				# print()
				# for i in range(3):
				# 	print(f"object_pose_noise_abs_avg/pos_{labels[i]} = ", object_pos_noise_abs_avg[i].item())
				# print()
				# for i in range(3):
				# 	print(f"object_pose_noise_abs_avg/rot_{labels[i]} = ", object_rot_noise_abs_avg[i].item())
				# print()
				# for i in range(16):
				# 	print(f"dof_pos_noise_abs_avg/dof_{i+1} = ", dof_pos_noise_abs_avg[i].item())
				# print()
				# for i in range(16):
				# 	print(f"action_noise_abs_avg/dof_{i+1} = ", action_noise_abs_avg[i].item())

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
			
	def realtime_plotter(self):
		if self.frame % 10 == 0:
			for ax in self.rt_plot_ax_obs_cube_pose.flat:
				ax.cla()
			for ax in self.rt_plot_ax_obs_dof.flat:
				ax.cla()
			for ax in self.rt_plot_ax_actions.flat:
				ax.cla()


			# print(self.rt_plot_frame_buffer.shape, self.rt_plot_obs_cube_pos_buffer.shape)
			self.rt_plot_frame_buffer = np.vstack([self.rt_plot_frame_buffer, self.frame])
			# print(np.array([self.cube_rot_noise_scaled.cpu().numpy(),]).shape, self.rt_plot_obs_cube_pos_buffer.shape)
			# print(np.array([self.cube_rot_noise_scaled.cpu().numpy(),]))
			# print(self.rt_plot_obs_cube_pos_buffer)
			self.rt_plot_obs_cube_pos_buffer = np.vstack([self.rt_plot_obs_cube_pos_buffer, np.array([self.cube_pos_noise_scaled.cpu().numpy(),])])
			self.rt_plot_obs_cube_rot_buffer = np.vstack([self.rt_plot_obs_cube_rot_buffer, np.array([self.cube_rot_noise_scaled.cpu().numpy(),])])
			self.rt_plot_actions_buffer = np.vstack([self.rt_plot_actions_buffer, np.array([self.action_noise_scaled.cpu().numpy(),])])
			self.rt_plot_obs_dof_buffer = np.vstack([self.rt_plot_obs_dof_buffer, np.array([self.dof_pos_noise_scaled.cpu().numpy(),])])
			# print(self.rt_plot_obs_cube_pos_buffer)
			# print(self.cube_pos_noise_scaled.shape, self.rt_plot_obs_cube_pos_buffer.shape)

			for i in range(self.num_envs):
				self.rt_plot_ax_obs_cube_pose[0,0].plot(self.rt_plot_frame_buffer[:], self.rt_plot_obs_cube_pos_buffer[:, i, 0])
				self.rt_plot_ax_obs_cube_pose[0,1].plot(self.rt_plot_frame_buffer[:], self.rt_plot_obs_cube_pos_buffer[:, i, 1])
				self.rt_plot_ax_obs_cube_pose[0,2].plot(self.rt_plot_frame_buffer[:], self.rt_plot_obs_cube_pos_buffer[:, i, 2])
				self.rt_plot_ax_obs_cube_pose[1,0].plot(self.rt_plot_frame_buffer[:], self.rt_plot_obs_cube_rot_buffer[:, i, 0])
				self.rt_plot_ax_obs_cube_pose[1,1].plot(self.rt_plot_frame_buffer[:], self.rt_plot_obs_cube_rot_buffer[:, i, 1])
				self.rt_plot_ax_obs_cube_pose[1,2].plot(self.rt_plot_frame_buffer[:], self.rt_plot_obs_cube_rot_buffer[:, i, 2])

				for j, ax in enumerate(self.rt_plot_ax_actions.flat):
					ax.plot(self.rt_plot_frame_buffer[:], self.rt_plot_actions_buffer[:, i, j])

				for j, ax in enumerate(self.rt_plot_ax_obs_dof.flat):
					ax.plot(self.rt_plot_frame_buffer[:], self.rt_plot_obs_dof_buffer[:, i, j])

			for i, ax in enumerate(self.rt_plot_ax_actions.flat):
				# ax.set(xlabel='frame')
				ax.set_title(f"dof_{i+1} Action Noise")

			for i, ax in enumerate(self.rt_plot_ax_obs_dof.flat):
				# ax.set(xlabel='frame')
				ax.set_title(f"dof_{i+1} Observation Noise")

			self.rt_plot_ax_obs_cube_pose[0,0].set_title("Cube Pos X Noise")					
			self.rt_plot_ax_obs_cube_pose[0,1].set_title("Cube Pos Y Noise")					
			self.rt_plot_ax_obs_cube_pose[0,2].set_title("Cube Pos Z Noise")					
			self.rt_plot_ax_obs_cube_pose[1,0].set_title("Cube Rot X Noise")					
			self.rt_plot_ax_obs_cube_pose[1,1].set_title("Cube Rot Y Noise")					
			self.rt_plot_ax_obs_cube_pose[1,2].set_title("Cube Rot Z Noise")	
			plt.pause(0.0001)

class AllegroHandDextremeAdversarialObservations(AllegroHandDextremeADR, AllegroHandDextreme, ADRVecTask):
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

		if self.use_dof_pos_obs:
			self.cfg["env"]["numActions"] = 6 + 16
		else:
			self.cfg["env"]["numActions"] = 6

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
		self.clip_cube_rot_noise = self.cfg["env"].get("clipCubeRotNoise", 0.1)		
		self.clip_cube_pos_noise = self.cfg["env"].get("clipCubePosNoise", 0.05)		
		box_boundary = np.concatenate((np.ones(3) * self.clip_cube_pos_noise, np.ones(3) * self.clip_cube_rot_noise))
		
		if self.use_dof_pos_obs:
			self.clip_dof_pos_noise = self.cfg["env"].get("clipDofPosNoise", 0.1)		
			box_boundary = np.concatenate((box_boundary, np.ones(16) * self.clip_dof_pos_noise))

		self.act_space = spaces.Box(-box_boundary, box_boundary)

		# self.num_actions = 16

	def get_num_obs_dict(self, num_dofs):
		num_obs_dict = AllegroHandDextremeADR.get_num_obs_dict(self, num_dofs)
		if self.use_dof_pos_obs:
			num_obs_dict["last_actions_full"] = 6 + 16
		else:
			num_obs_dict["last_actions_full"] = 6			
		num_obs_dict["last_actions"] = 16
		return num_obs_dict

	def _read_cfg(self):
		AllegroHandDextremeADR._read_cfg(self)
		self.base_controller_checkpoint = self.cfg["onnx_model_checkpoint"]
		self.use_dof_pos_obs = self.cfg.get("useDofPosObs", True)

	def _init_pre_sim_buffers(self):
		AllegroHandDextremeADR._init_pre_sim_buffers(self)
		self.affine_actions_additive = torch.zeros(self.cfg["env"]["numEnvs"], 16, dtype=torch.float, device=self.sim_device)
		self.affine_actions_scaling = torch.zeros(self.cfg["env"]["numEnvs"], 16, dtype=torch.float, device=self.sim_device)

	def _init_post_sim_buffers(self):
		AllegroHandDextremeADR._init_post_sim_buffers(self)
		self.prev_actions = torch.zeros(self.num_envs, 16, dtype=torch.float, device=self.device)
		self.prev_actions_queue = torch.zeros(self.cfg["env"]["numEnvs"], self.action_latency_max + 1, 16, dtype=torch.float, device=self.sim_device)
		
		self.base_controller = BaseControllerPlugin(self.base_controller_checkpoint, self.device)

	def pre_physics_step(self, obs_noises):
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

		self.cube_pos_noise_scaled = obs_noises[:,0:3]
		self.cube_rot_noise_scaled = obs_noises[:,3:6]
		if self.use_dof_pos_obs:
			self.dof_pos_noise_scaled = obs_noises[:,6:22]
		self.add_noise_to_observations()
		self.base_controller_actions = self.base_controller.get_action(self.obs_dict)
		actions = self.base_controller_actions

		self.actions_full = obs_noises.clone().to(self.device)

		self.apply_actions(actions)
		self.apply_random_forces()

	def add_noise_to_observations(self):
		noisy_cube_pos = self.obs_dict["object_pose_cam_randomized"][:, 0:3] + self.cube_pos_noise_scaled

		cube_rot_noise_quat = quat_from_euler_xyz(self.cube_rot_noise_scaled[:, 0], 
												  self.cube_rot_noise_scaled[:, 1],
												  self.cube_rot_noise_scaled[:, 2])
		
		cube_rot = self.obs_dict["object_pose_cam_randomized"][:, 3:7]
		noisy_cube_rot = quat_mul(cube_rot, cube_rot_noise_quat)

		quat_diff = quat_mul(cube_rot, quat_conjugate(noisy_cube_rot))
		rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
		self.object_rot_noise = rot_dist

		noisy_cube_pose_obs = torch.cat((noisy_cube_pos, noisy_cube_rot), axis=-1)

		self.obs_dict["object_pose_cam_randomized"] = noisy_cube_pose_obs
		if self.use_dof_pos_obs:
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
				if self.use_dof_pos_obs:
					dof_pos_noise_avg = self.dof_pos_noise_scaled.mean(dim=1)
					for i in range(16):
						self.eval_summaries.add_scalar(f"dof_pos_noise_avg/dof_{i+1}", dof_pos_noise_avg[i].item(), self.frame)
				
				object_rot_noise_avg = self.cube_rot_noise_scaled.mean(dim=1)
				object_pos_noise_avg = self.cube_pos_noise_scaled.mean(dim=1)
				object_rot_noise_avg_angle = self.object_rot_noise.mean()
				
				self.eval_summaries.add_scalar(f"object_rot_noise_angle", object_rot_noise_avg_angle.item(), self.frame)

				labels = ["x", "y", "z"]
				for i in range(3):
					self.eval_summaries.add_scalar(f"object_pose_noise_avg/pos_{labels[i]}", object_pos_noise_avg[i].item(), self.frame)
					self.eval_summaries.add_scalar(f"object_pose_noise_avg/rot_{labels[i]}", object_rot_noise_avg[i].item(), self.frame)


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
	success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
	fall_penalty: float, max_consecutive_successes: int, av_factor: float, num_success_hold_steps: int
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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

	# Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
	reward = dist_rew + rot_rew + action_penalty + action_delta_penalty + velocity_penalty + reach_goal_rew + fall_rew + timeout_rew + \
			 action_noise_penalty + cube_pos_noise_penalty + cube_rot_noise_penalty + dof_pos_noise_penalty

	num_resets = torch.sum(resets)
	finished_cons_successes = torch.sum(successes * resets.float())

	cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

	return reward, resets, goal_resets, progress_buf, hold_count_buf, successes, cons_successes, \
		dist_rew, rot_rew, action_penalty, action_delta_penalty, velocity_penalty, reach_goal_rew, fall_rew, timeout_rew  # return individual rewards for visualization
