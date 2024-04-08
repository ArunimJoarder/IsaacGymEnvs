import os, argparse
import shutil

main_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def copy_checkpoint(src_dir: str, dst_dir: str, exp: str, model_num: int):
	latest_edited_file = max([f for f in os.scandir(src_dir)], key=lambda x: x.stat().st_mtime).name
	checkpoint = os.path.join(src_dir, latest_edited_file)
	print("\n\n\nSelected checkpoint:", checkpoint, "\n\n\n")
	os.makedirs(dst_dir, exist_ok=True)
	dst = os.path.join(dst_dir, exp + "_" + str(model_num) + ".pth")
	shutil.copy2(checkpoint, dst)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-e", "--experiment", type=str, help = "Name of experiment", required=True)
	parser.add_argument("-m", "--model_number", type=int, help="Identifier of model", default=0)
	parser.add_argument("-d", "--dst_dir", type=str, help = "Destination directory", default="runs/default_ensemble")
	args = parser.parse_args()
	
	# print("main_dir:", main_dir)

	dst_dir = os.path.join(main_dir, args.dst_dir)
	# print("dst_dir:", dst_dir)
	
	nn_dir = os.path.join(args.experiment + "_model_" + str(args.model_number), "nn")
	exp_dir = os.path.join(main_dir, "runs", nn_dir)
	if not os.path.exists(exp_dir):
		msg = "\"" + nn_dir + "\" does not exist in " + os.path.join(main_dir, "runs")
		raise Exception(msg)
	# print("exp_dir:", exp_dir)

	copy_checkpoint(exp_dir, dst_dir, args.experiment, args.model_number)