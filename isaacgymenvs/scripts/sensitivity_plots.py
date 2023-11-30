import argparse
from numpy.typing import NDArray
from typing import List, Dict, Tuple

import numpy as np
import os
from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pyplot as plt
import pypdf

eval_summaries_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_summaries")

base_data: Dict[str, Tuple[float, float]] = {}

def get_data(tags: List[str], exp_name: str, prop_name: str) -> Dict[str, Dict[str, NDArray[np.float16]]]:
	experiment_dir = os.path.join(eval_summaries_path, exp_name, prop_name)

	data: Dict[str, Dict[str, NDArray[np.float16]]] = {}
	for tag in tags:
		data[tag] = {}

	for subdir, _, files in os.walk(experiment_dir):
		for file in files:
			if file.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
				pass
			else:
				log_file = os.path.join(subdir, file)
				dir_name = os.path.basename(os.path.dirname(log_file)).replace("adr.params.", "")

				temp_data: Dict[str, List[float]] = {}
				for tag in tags:
					temp_data[tag] = []

				for event in summary_iterator(log_file):
					if event.step >= 6000:
						for value in event.summary.value:
							if value.tag in tags:
								temp_data[value.tag].append(value.simple_value)

				for tag in tags:
					data[tag][dir_name] = np.array(temp_data[tag])

	for tag in tags:
		data[tag] = dict(sorted(data[tag].items(), key = lambda x: x[0]))

	return data

def plot_boxplot(data: Dict[str, NDArray[np.float16]], data_label: str, exp_name: str, prop_name: str):
	slash_id = data_label.find("/") + 1 
	data_label = data_label[slash_id:]

	experiment_dir = os.path.join(eval_summaries_path, exp_name, prop_name)

	fig, ax = plt.subplots()
	fig.set_size_inches(13.6, 13.6/4)
	ax.set_title(prop_name)
	ax.boxplot(list(data.values()), labels=list(data.keys()), vert=False)
	
	# if data_label == "consecutive_successes":
	# 	plt.xlim(0, 45)
	# elif data_label == "last_ep_successes":
	# 	plt.xlim(0, 45)
	# elif data_label == "avg_success_per_minute":
	# 	plt.xlim(0, 30)
	# else:
	# 	plt.xlim(0, 45)
	
	references = [base_data[data_label][0] - 3*base_data[data_label][1], base_data[data_label][0], base_data[data_label][0] + 3*base_data[data_label][1]]
	# references = [base_data[data_label][0]*0.85, base_data[data_label][0], base_data[data_label][0]*1.15]

	plt.xlim(0, references[2])
	if data_label == "avg_success_time_seconds":
		plt.xlim(0, 25)
	ymin, ymax = plt.ylim()

	plt.vlines(references,ymin = ymin, ymax = ymax, color=['r', 'g', 'r'], linestyles=['dotted', 'solid', 'dotted'])
	plt.xlabel(data_label.replace("_", " ").capitalize())

	plt.xticks(list(plt.xticks()[0]) + [references[1]])
	plotname_pdf = os.path.join(experiment_dir, data_label + "_summary.pdf")
	plt.savefig(plotname_pdf, bbox_inches="tight")
	plotname_png = os.path.join(experiment_dir, data_label + "_summary.png")
	plt.savefig(plotname_png, bbox_inches="tight")
	plt.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-t", "--tags", nargs='+', type=str, help = "Tags to look for in the tfevent file", default=["consecutive_successes", "last_ep_successes", "policy_speed/avg_success_per_minute", "policy_speed/avg_success_time_seconds"])
	parser.add_argument("-e", "--experiment_name", type=str, help = "Name of experiment series", default="first_exp")
	args = parser.parse_args()


	experiment_dir = os.path.join(eval_summaries_path, args.experiment_name)

	base_dataset = get_data(args.tags, args.experiment_name, "base")
	for tag, data in base_dataset.items():
		all_data = np.hstack(list(data.values()))
		slash_id = tag.find("/") + 1 
		tag = tag[slash_id:]
		base_data[tag] = all_data.mean(), np.sqrt(all_data.var())

	for prop_name in os.listdir(experiment_dir):
		if not os.path.isfile(os.path.join(experiment_dir, prop_name)):
			dataset = get_data(args.tags, args.experiment_name, prop_name)
			for tag, data in dataset.items():
				plot_boxplot(data, tag, args.experiment_name, prop_name)

	merge_tags = set(["consecutive_successes", "last_ep_successes", "avg_success_per_minute", "avg_success_time_seconds"])
	merge_prop_set = None
	# merge_prop_set = set(["action_latency", "hand_damping", "hand_stiffness", "affine_action_white", "affine_action_scaling", "affine_action_additive", "affine_dof_pos_white", "affine_dof_pos_scaling", "affine_dof_pos_additive", "affine_cube_pose_scaling", "affine_cube_pose_white", "affine_cube_pose_additive", "hand_upper", "hand_lower"])

	mergers = {}
	for tag in merge_tags:
		mergers[tag] = pypdf.PdfMerger()

	for prop_name in sorted(os.listdir(experiment_dir)):
		if not os.path.isfile(os.path.join(experiment_dir, prop_name)):
			if merge_prop_set is None or prop_name in merge_prop_set:
				subdir = os.path.join(experiment_dir, prop_name)
				for pdf in os.listdir(subdir):
					filename = os.path.join(subdir, pdf)
					if os.path.isfile(filename):
						if pdf.endswith(".pdf"):
							tag = pdf.replace("_summary.pdf", "")
							if tag in merge_tags:
								mergers[tag].append(filename)

	for tag in merge_tags:
		mergers[tag].write(os.path.join(experiment_dir, tag + "_compartive_summary.pdf"))
		mergers[tag].close()