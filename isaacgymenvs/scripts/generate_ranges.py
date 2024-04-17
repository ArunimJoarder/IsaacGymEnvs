from typing import Optional, List, Tuple, Union
from numpy.typing import NDArray
import numpy as np
import os, argparse

ADR_BASE_INIT =   { "hand_damping"				: [ 0.50, 2.00],
					"hand_stiffness"			: [ 0.80, 1.20],
					"hand_joint_friction"		: [ 0.80, 1.20],
					"hand_armature"				: [ 0.80, 1.20],
					"hand_effort"				: [ 0.90, 1.10],
					"hand_lower"				: [ 0.00, 0.00],
					"hand_upper"				: [ 0.00, 0.00],
					"hand_mass"					: [ 0.80, 1.20],
					"hand_friction_fingertips"	: [ 0.90, 1.10],
					"object_mass"				: [ 0.80, 1.20],
					"object_friction"			: [ 0.40, 0.80],
					"action_delay_prob"			: [ 0.00, 0.05],
					"action_latency"			: [ 0.00, 0.00],
					"affine_action_additive"	: [ 0.00, 0.04],
					"affine_action_white"		: [ 0.00, 0.04],
					"affine_cube_pose_additive"	: [ 0.00, 0.04],
					"affine_cube_pose_white"	: [ 0.00, 0.04],
					"affine_dof_pos_additive"	: [ 0.00, 0.04],
					"affine_dof_pos_white"		: [ 0.00, 0.04],
					"rna_alpha"					: [ 0.00, 0.00]}

ADR_BASE_RANGES = { "hand_damping"				: [ 0.32, 2.76],
					"hand_stiffness"			: [ 0.31, 1.60],
					"hand_joint_friction"		: [ 0.00, 2.38],
					"hand_armature"				: [ 0.01, 1.50],
					"hand_effort"				: [ 0.68, 2.71],
					"hand_lower"				: [-2.76, 0.32],
					"hand_upper"				: [-0.24, 2.70],
					"hand_mass"					: [ 0.01, 2.38],
					"hand_friction_fingertips"	: [ 0.10, 2.00],
					"object_mass"				: [ 0.18, 1.53],
					"object_friction"			: [ 0.22, 2.00],
					"action_delay_prob"			: [ 0.00, 0.59],
					"action_latency"			: [ 0.00, 1.70],
					"affine_action_additive"	: [ 0.00, 0.37],
					"affine_action_white"		: [ 0.00, 0.56],
					"affine_cube_pose_additive"	: [ 0.00, 0.19],
					"affine_cube_pose_white"	: [ 0.00, 0.23],
					"affine_dof_pos_additive"	: [ 0.00, 0.28],
					"affine_dof_pos_white"		: [ 0.00, 0.31],
					"rna_alpha"					: [ 0.00, 0.19]}

ADR_BASE_LIMITS = { "hand_damping"				: [0.01, 20.0],
					"hand_stiffness"			: [0.01, 20.0],
					"hand_joint_friction"		: [0.00, 10.0],
					"hand_armature"				: [0.00, 10.0],
					"hand_effort"				: [0.40, 10.0],
					"hand_lower"				: [-5.0, 5.00],
					"hand_upper"				: [-5.0, 5.00],
					"hand_mass"					: [0.01, 10.0],
					"hand_friction_fingertips"	: [0.01, 2.00],
					"object_mass"				: [0.01, 10.0],
					"object_friction"			: [0.01, 2.00],
					"action_delay_prob"			: [0.00, 0.70],
					"action_latency"			: [0.00, 60.0],
					"affine_action_additive"	: [0.00, 4.00],
					"affine_action_white"		: [0.00, 4.00],
					"affine_cube_pose_additive"	: [0.00, 4.00],
					"affine_cube_pose_white"	: [0.00, 4.00],
					"affine_dof_pos_additive"	: [0.00, 4.00],
					"affine_dof_pos_white"		: [0.00, 4.00],
					"rna_alpha"					: [0.00, 1.00]}

def gen_new_range(extension_ratio: Union[float, List[float]] = 0.0,
				  param_names: Optional[List[str]] = None) -> Tuple[List[str], NDArray]:

	if param_names is None or param_names[0] == "A":
		print("Using all parameters")
		values = np.array([*ADR_BASE_RANGES.values()])
		limits = np.array([*ADR_BASE_LIMITS.values()])
		names  = [*ADR_BASE_RANGES.keys()]
	else:
		names  = [param_name for param_name in param_names if param_name in ADR_BASE_RANGES.keys()]
		values = np.array([ADR_BASE_RANGES[param_name] for param_name in names])
		limits = np.array([ADR_BASE_LIMITS[param_name] for param_name in names])

	if isinstance(extension_ratio, float):
		r = np.ones(values.shape[0]) * extension_ratio
	else:
		r = np.array(extension_ratio)

	assert values.shape[0] == limits.shape[0], "Number of limits should match number of params"
	assert r.shape[0] == values.shape[0], "Range extension does not match number of params"
	
	val_eq_lim_ind = [i for i in range(values.shape[0]) if np.all(values[i,:] == limits[i,:])]
	# for ind in val_eq_lim_ind:
	# 	print(names[ind])
	values = np.delete(values, val_eq_lim_ind, axis = 0)
	names = [names[i] for i in range(len(names)) if i not in val_eq_lim_ind]

	new_values = values.copy()

	diff = values[:,1] - values[:,0]
	lim_diff = limits[:,1] - limits[:,0]
	R = r * diff / (lim_diff - diff + 1e-15)
	new_values[:,0] = np.clip(values[:,0] + R * (limits[:,0] - values[:,0]), a_min=limits[:,0], a_max=None)
	new_values[:,1] = np.clip(values[:,1] + R * (limits[:,1] - values[:,1]), a_max=limits[:,1], a_min=None)

	# mean = (values[:,1] + values[:,0]) * 0.5
	# diff = (values[:,1] - values[:,0]) * 0.5
	# new_values[:,0] = mean - (1 + r) * diff
	# new_values[:,1] = mean + (1 + r) * diff
	for i in range(len(names)):
		if "affine" in names[i]:
			new_values[i,0] = values[i,0]
			new_values[i,1] = min([values[i,1] * (1 + r[i]), limits[i,1]])
	# 	elif names[i] in ["hand_joint_friction", "hand_mass", "action_delay_prob", "action_latency", "rna_alpha"]:
	# 		new_values[i,0] = values[i,0]
	# 		new_values[i,1] = min([values[i,0] + 2 * (1 + r[i]) * diff[i], limits[i,1]])
	# 	elif names[i] in ["hand_friction_fingertips", "object_friction"]:
	# 		new_values[i,1] = values[i,1]
	# 		new_values[i,0] = max([values[i,1] - 2 * (1 + r[i]) * diff[i], limits[i,0]])
	# new_diff = (new_values[:,1] - new_values[:,0]) * 0.5

	# left_limit_reached_ind  = list((new_values[:,0] < limits[:,0]).nonzero()[0])
	# right_limit_reached_ind = list((new_values[:,1] > limits[:,1]).nonzero()[0])
	
	# for i in left_limit_reached_ind:
	# 	if i in right_limit_reached_ind:
	# 		new_values[i,0] = limits[i,0]
	# 		new_values[i,1] = limits[i,1]
	# 	else:
	# 		new_values[i,0] = limits[i,0]
	# 		new_values[i,1] = min([new_values[i,0] + 2 * new_diff[i], limits[i,1]])
	# for i in right_limit_reached_ind:
	# 	if not i in left_limit_reached_ind:
	# 		new_values[i,1] = limits[i,1]
	# 		new_values[i,0] = max([new_values[i,1] - 2 * new_diff[i], limits[i,0]])

	new_diff = (new_values[:,1] - new_values[:,0])
	e_ratio = new_diff / (diff + 1e-5) - 1

	e_ratio = np.round(e_ratio, decimals = 3)	
	new_values = np.round(new_values, decimals = 2)	
	return names, new_values, e_ratio

def print_to_file(names:  List[str],
				  values: NDArray,
				  e_ratio: NDArray,
				  filename: Optional[str] = None):
	assert len(names) == values.shape[0], "Number of keys is not equal to number of ranges"
	
	if filename is None:
		filename = "sensitivity_ranges.txt"
	if filename[-4:] != ".txt":
		filename += ".txt"
	filedir = os.path.dirname(os.path.abspath(__file__))
	subdir = "ranges"
	filename = os.path.join(filedir, subdir, filename)

	with open(filename, "w") as f:
		for i in range(len(names) - 1):
			f.write(names[i] + " ")
		f.write(names[-1] + "\n")
		for i in range(len(names) - 1):
			f.write(f"{values[i,0]:.3f}" + " ")
		f.write(f"{values[-1,0]:.3f}\n")
		for i in range(len(names) - 1):
			f.write(f"{values[i,1]:.3f}" + " ")
		f.write(f"{values[-1,1]:.3f}\n")
		for i in range(len(names) - 1):
			f.write(f"{e_ratio[i]:.2f}" + " ")
		f.write(f"{e_ratio[-1]:.2f}\n")
		# for i in range(len(names)):
		# 	f.write(names[i].ljust(30) + f"{e_ratio[i]:.2f}".rjust(5) + "\t\t[" + f"{values[i,0]:.3f},".rjust(7) + f"{values[i,1]:.3f}".rjust(6) + "]" + "\n")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("-r", "--extension_ratio", nargs='+', type=float, help = "Extension ratio for range (Can be float or list of floats)")
	parser.add_argument("-p", "--param_names", nargs='+', type=str, help = "List of params to generate ranges for")
	parser.add_argument("-f", "--filename", help = "Name of file to which ranges are saved")
	
	args = parser.parse_args()

	r = args.extension_ratio
	if r is not None:
		if len(r) == 1:
			r = r[0]
	else:
		r = 0.0

	n, v, e = gen_new_range(r, args.param_names)

	print_to_file(n, v, e, filename = args.filename)
