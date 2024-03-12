import argparse
from numpy.typing import NDArray
from typing import List, Dict, Tuple, Optional

import numpy as np
import os, sys
from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pyplot as plt

eval_summaries_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_summaries")

def get_data(tags: List[str], exp_name: str) -> Dict[str, Dict[str, NDArray[np.float16]]]:
	experiment_dir = os.path.join(eval_summaries_path, exp_name)

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
		data[tag] = dict(sorted(data[tag].items(), key = lambda x: x[1].mean(), reverse = True))

	return data

def analyze(dataset: Dict[str, Dict[str, NDArray[np.float16]]],
			print_flag: bool = True,
			main_tag: Optional[str] = None) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
	avg: Dict[str, Dict[str, float]] = {}
	var: Dict[str, Dict[str, float]] = {}

	for tag, data in dataset.items():
		avg_tag: Dict[str, float] = {}
		var_tag: Dict[str, float] = {}
		for prop in data.keys():
			avg_tag[prop] = data[prop].mean()
			var_tag[prop] = data[prop].var()

		delta_avg: Dict[str, float] = {}
		for prop in data.keys():
			# delta_avg[prop] = (avg_tag[prop] - avg_tag["base"])/avg_tag["base"]*100
			delta_avg[prop] = avg_tag[prop]

		delta_avg = dict(sorted(delta_avg.items(), key = lambda x: x[1]))
		var_tag = dict(sorted(var_tag.items(), key = lambda x: x[1]))

		avg[tag] = delta_avg
		var[tag] = var_tag

	if print_flag:
		prop_space = 155
		topline = "".center(prop_space + 6, "_")
		headline = "||" + "".center(prop_space + 6) + "||"
		fillline = "||" + "".center(prop_space + 6, "-") + "||"
		subline = "||" + "Property".center(prop_space + 6) + "||"
		for tag in dataset.keys():
			slash_id = tag.find("/") + 1 
			tag = tag[slash_id:]
			topline += "".center(36, "_")
			headline += tag.center(33) + "||"
			fillline += "".center(33, "-") + "||"
			subline += "mean".center(16) + "|" + "std".center(16) + "||"

		print(topline, headline, fillline, subline, fillline, sep="\n")

		if main_tag is None:
			main_tag = list(dataset.keys())[0]

		for j, prop in enumerate(avg[main_tag].keys(),1):
			prop_label = prop
			# for i in "0123456789.-":
			# 	prop_label = prop_label.replace(i,"")
			# prop_label = prop_label.replace("__", "")
			prop_line = f"|| {j:3d}) " + prop_label.ljust(prop_space) + "||"

			for tag in dataset.keys():
				prop_line += f"{avg[tag][prop]:.2f}".rjust(10) + "".center(6) + "|" + f"{np.sqrt(var[tag][prop]):.5f}".rjust(12) + "".center(4) + "||"
			print(prop_line)
		
		fillline = "||" + "".center(prop_space + 6, "_") + "||"
		for tag in dataset.keys():
			fillline += "".center(16, "_") + "|" + "".center(16, "_") + "||"
		print(fillline, "\n")

	return avg, var

def plot_boxplot(data: Dict[str, NDArray[np.float16]], data_label: str, exp_name: str):
	slash_id = data_label.find("/") + 1 
	data_label = data_label[slash_id:]

	experiment_dir = os.path.join(eval_summaries_path, exp_name)

	fig, ax = plt.subplots()
	fig.set_size_inches(13.6, 13.6)
	ax.set_title(data_label.replace("_", " ").capitalize())
	ax.boxplot(list(data.values()), labels=list(data.keys()), vert=False)
	plt.xlabel(data_label.replace("_", " ").capitalize())
	plt.autoscale()
	plotname = os.path.join(experiment_dir, data_label + "_summary.pdf")
	plt.savefig(plotname, bbox_inches="tight")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-t", "--tags", nargs='+', type=str, help = "Tags to look for in the tfevent file", default=["consecutive_successes", "last_ep_successes", "policy_speed/avg_success_per_minute", "policy_speed/avg_success_time_seconds"])
	parser.add_argument("-T", "--main_tag", type=str, help = "Tag wrt which bad performance is measured", default="consecutive_successes")
	parser.add_argument("-V", "--fault_value", type=float, help = "Value below which performance drop is considered bad", default=15.0)
	parser.add_argument("-e", "--experiment_name", type=str, help = "Name of experiment series", default="first_exp")
	parser.add_argument("-p","--plot", action="store_true", help = "If you want to plot data")
	parser.add_argument("-np","--no_print", action="store_false", help = "If you don't want to print analysis")
	args = parser.parse_args()

	assert args.main_tag in args.tags, "List of tags should contain main_tag"

	dataset = get_data(args.tags, args.experiment_name)

	if args.plot:
		for tag, data in dataset.items():
			plot_boxplot(data, tag, args.experiment_name)
	
	averages, _ = analyze(dataset, args.no_print, args.main_tag)

	main_tag_averages = averages[args.main_tag]
	fault_props = [prop for prop, avg in main_tag_averages.items() if (avg - main_tag_averages["base"])/main_tag_averages["base"]*100 <= -args.fault_value]
	
	print(fault_props)
	for i in range(len(fault_props)):
		prop_label = fault_props[i]
		for i in "0123456789.-":
			prop_label = prop_label.replace(i,"")
		prop_label = prop_label.replace(" ", "")
		prop_label = prop_label.replace("__", "")
		print(prop_label, end=" ")
	print()