from typing import Optional, List, Tuple, Union
from numpy.typing import NDArray
import numpy as np
import os, argparse

ADR_BASE_RANGES = { "hand_damping"				: [ 0.32, 2.76],
					"hand_stiffness"			: [ 0.31, 1.60],
					"hand_joint_friction"		: [ 0.00, 2.38],
					"hand_armature"				: [ 0.01, 1.50],
					"hand_effort"				: [ 0.68, 2.71],
					"hand_lower"				: [-2.76, 0.32],
					"hand_upper"				: [-0.24, 2.70],
					"hand_mass"					: [ 0.01, 2.38],
					"hand_friction_fingertips"	: [ 0.10, 2.00],
					"hand_restitution"			: [ 0.00, 1.00],
					"object_mass"				: [ 0.18, 1.53],
					"object_friction"			: [ 0.22, 2.00],
					"object_restitution"		: [ 0.00, 1.00],
					"cube_obs_delay_prob"		: [ 0.00, 0.70],
					"cube_pose_refresh_rate"	: [ 1.00, 6.00],
					"action_delay_prob"			: [ 0.00, 0.59],
					"action_latency"			: [ 0.00, 1.70],
					"affine_action_scaling"		: [ 0.00, 0.00],
					"affine_action_additive"	: [ 0.00, 0.37],
					"affine_action_white"		: [ 0.00, 0.56],
					"affine_cube_pose_scaling"	: [ 0.00, 0.00],
					"affine_cube_pose_additive"	: [ 0.00, 0.19],
					"affine_cube_pose_white"	: [ 0.00, 0.23],
					"affine_dof_pos_scaling"	: [ 0.00, 0.00],
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
					"hand_restitution"			: [0.00, 1.00],
					"object_mass"				: [0.01, 10.0],
					"object_friction"			: [0.01, 2.00],
					"object_restitution"		: [0.00, 1.00],
					"cube_obs_delay_prob"		: [0.00, 0.70],
					"cube_pose_refresh_rate"	: [1.00, 6.00],
					"action_delay_prob"			: [0.00, 0.70],
					"action_latency"			: [0.00, 60.0],
					"affine_action_scaling"		: [0.00, 4.00],
					"affine_action_additive"	: [0.00, 4.00],
					"affine_action_white"		: [0.00, 4.00],
					"affine_cube_pose_scaling"	: [0.00, 4.00],
					"affine_cube_pose_additive"	: [0.00, 4.00],
					"affine_cube_pose_white"	: [0.00, 4.00],
					"affine_dof_pos_scaling"	: [0.00, 4.00],
					"affine_dof_pos_additive"	: [0.00, 4.00],
					"affine_dof_pos_white"		: [0.00, 4.00],
					"rna_alpha"					: [0.00, 1.00]}

def gen_new_range(extension_ratio_r: Union[float, List[float]] = 0.0,
				  extension_ratio_l: Optional[Union[float, List[float]]] = None,
				  param_names: Optional[List[str]] = None) -> Tuple[List[str], NDArray]:

	if extension_ratio_l is None:
		extension_ratio_l = extension_ratio_r

	if param_names is None or param_names[0] == "A":
		print("Using all parameters")
		values = np.array([*ADR_BASE_RANGES.values()])
		limits = np.array([*ADR_BASE_LIMITS.values()])
		names  = [*ADR_BASE_RANGES.keys()]
	else:
		names  = [param_name for param_name in param_names if param_name in ADR_BASE_RANGES.keys()]
		values = np.array([ADR_BASE_RANGES[param_name] for param_name in names])
		limits = np.array([ADR_BASE_LIMITS[param_name] for param_name in names])

	if isinstance(extension_ratio_r, float):
		er_r = np.ones(values.shape[0])*extension_ratio_r
	else:
		er_r = np.array(extension_ratio_r)

	if isinstance(extension_ratio_l, float):
		er_l = np.ones(values.shape[0])*extension_ratio_l
	else:
		er_l = np.array(extension_ratio_l)

	assert values.shape[0] == limits.shape[0], "Number of limits should match number of params"
	assert er_l.shape[0] == values.shape[0], "Range extension on left does not match number of params"
	assert er_r.shape[0] == values.shape[0], "Range extension on right does not match number of params"
	
	range_mag_l = np.abs(limits[:,0] - values[:,0])
	range_mag_r = np.abs(limits[:,1] - values[:,1])
	values[:,0] = np.clip(values[:,0] - er_l*range_mag_l, limits[:,0], limits[:,1])
	values[:,1] = np.clip(values[:,1] + er_r*range_mag_r, limits[:,0], limits[:,1])

	val_eq_lim_ind = [i for i in range(values.shape[0]) if np.all(values[i,:] == limits[i,:])]
	values = np.delete(values, val_eq_lim_ind, axis = 0)
	names = [names[i] for i in range(len(names)) if i not in val_eq_lim_ind]

	values = np.round(values, decimals = 3)	
	return names, values

def print_to_file(names:  List[str],
				  values: NDArray,
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
		
		

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("-l", "--left_extension", nargs='+', type=float, help = "Extension ratio for left limit of range (Can be float or list of floats)")
	parser.add_argument("-r", "--right_extension", nargs='+', type=float, help = "Extension ratio for left limit of range (Can be float or list of floats)")
	parser.add_argument("-p", "--param_names", nargs='+', type=str, help = "List of params to generate ranges for")
	parser.add_argument("-f", "--filename", help = "Name of to which ranges are saved")
	
	args = parser.parse_args()

	re = args.right_extension
	if re is not None:
		if len(re) == 1:
			re = re[0]
	else:
		re = 0.0

	le = args.left_extension
	if le is not None:
		if len(le) == 1:
			le = le[0]

	n, v = gen_new_range(re, le, args.param_names)

	print_to_file(n, v, filename = args.filename)
