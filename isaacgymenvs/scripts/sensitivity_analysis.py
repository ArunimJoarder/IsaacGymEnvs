import os
import click

import pandas as pd

import matplotlib.pyplot as plt
plt.style.use(os.path.abspath(os.path.join(os.path.dirname(__file__), "presentation.mplstyle")))

import matplotlib.cm as mplcm
import matplotlib.colors as colors
import numpy as np

NUM_COLORS = 15

# plt.rcParams['savefig.pad_inches'] = 0

cm = plt.get_cmap('tab20')
cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)

def plot_lines(df: pd.DataFrame, chkpt: str, metric: str):
    fig, ax = plt.subplots()

    ax.set_title("Sensitivity Analysis - " + chkpt)
    ax.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(NUM_COLORS)])

    labels = []
    for param in df.index.get_level_values(0).unique():
        if param == "base":
            continue
        p_df = df[param]
        p_df.plot(ax=ax, marker='.', ms=40, linewidth=10)
        labels.append(param)

    ax.set(ylabel="\% change in " + metric)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height*0.9])

    # Put a legend to the right of the current axis
    ax.legend(labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize="44")

    # figsize: 18.5, 11
    # plt.show()
    save_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "images")
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.join(save_dir, "sensitivity_analysis_" + chkpt + "_" + metric + ".png")
    plt.savefig(filename)

@click.command()
@click.argument("df_file")
@click.option("--steady-state-cutoff", "-s", help="steps after which model stabilizes", default=6000)
def main(df_file: str, steady_state_cutoff: int):
    df = pd.read_csv(df_file)
    ignore_params = ["rna_alpha", "hand_armature", "hand_effort", "hand_mass", "object_mass", "object_friction", "hand_friction_fingertips", "action_delay_prob"]

    for i_param in ignore_params:
        mask = df["param"] == i_param
        df = df[~mask]


    steady_state_df = df[df["step"] > steady_state_cutoff]
 
    for metric in ["consecutive_successes", "last_ep_successes"]:
        metric_ss_df = steady_state_df[steady_state_df["metric"] == metric]

        mean_df = metric_ss_df.groupby(['checkpoint', 'param', 'ratio'])['value'].mean().sort_index()

        for chkpt in mean_df.index.get_level_values(0).unique():
            chkpt_mean_df: pd.Series = mean_df[chkpt]
            chkpt_mean_df = (chkpt_mean_df - chkpt_mean_df["base"][0.0]) / chkpt_mean_df["base"][0.0] * 100

            if chkpt == "base_chkpt":
                chkpt = "base_controller"
            plot_lines(chkpt_mean_df, chkpt, metric)

if __name__ == "__main__":
    main()