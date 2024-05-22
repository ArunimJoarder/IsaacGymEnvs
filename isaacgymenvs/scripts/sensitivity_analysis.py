import os
import click

import pandas as pd

import matplotlib.pyplot as plt
plt.style.use(os.path.abspath(os.path.join(os.path.dirname(__file__), "presentation.mplstyle")))

import matplotlib.cm as mplcm
import matplotlib.colors as colors
from matplotlib.ticker import MultipleLocator
import numpy as np

NUM_COLORS = 15

# plt.rcParams['savefig.pad_inches'] = 0

cm = plt.get_cmap('tab20')
cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)

def plot_lines(df, chkpt: str, metric: str, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    words = chkpt.split('_')
    title = ""
    for word in words:
        title += word.capitalize() + " "
    ax.set_title("Sensitivity Analysis - " + title)
    ax.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(NUM_COLORS)])

    labels = []
    for param in df.index.get_level_values(0).unique():
        if param == "base" or param == "Action parameters":
            continue
        p_df = df[param]
        p_df.plot(ax=ax, marker='.', ms=40, linewidth=10)
        labels.append(param)

    ax.set(ylabel="\% change in avg_" + metric)
    ax.tick_params(width=3, length=20)
    ax.set_ylim(-80, 5)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.16, box.width, box.height*0.9])

    # Put a legend to the right of the current axis
    ax.legend(labels, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4)
    # ax.legend(labels)
    plt.grid(visible=True, which='major', axis='y', linewidth=5, linestyle='-')

    # figsize: 18.5, 11
    # plt.show()
    save_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "images")
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.join(save_dir, "sensitivity_analysis_" + chkpt + "_" + metric + ".png")
    # filename = os.path.join(save_dir, "sensitivity_analysis_" + chkpt + "_" + metric + "_presentation.png")
    plt.savefig(filename)
    return ax

def plot_regions(fdf, chkpt: str, metric: str, i = 0, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    ax.set_axisbelow(True)

    colors = ['tab:orange', 'tab:green',]
    linecolors = ['orangered', 'darkgreen',]
    alphas = 0.1
    titles = ['Base Controller', 'Improved Controller']

    ax.set_title("Sensitivity Analysis - " + titles[i])

    # # for chkpt in fdf.index.get_level_values(0).unique():
    # for chkpt in ["base_chkpt", "ft_delta_ensemble_adv_prob_0.30_value_selection"]:
    #     df: pd.Series = fdf[chkpt]
    #     df = (df - df["base"][0.0]) / df["base"][0.0] * 100
    #     temp = [*fdf.index.get_level_values(1).unique()]
    #     temp.remove("base")
    #     multi_index = pd.MultiIndex.from_product([temp, [0.0]], names=['param', 'ratio'])
    #     base_value = df["base", 0.0]
    #     zeros_series = pd.Series(np.ones((len(temp),)) * base_value, index=multi_index)

    #     df = pd.concat([df, zeros_series]).sort_index()
    #     df.name = "mean"
    #     for param in df.index.get_level_values(0).unique():
    #     # for param in ["hand_lower"]:
    #         if param == "base" or param == "Action parameters":
    #             continue
    #         p_df = df[param]
    #         ax.fill_between(p_df.index.get_level_values(0), p_df.to_numpy(), 0, facecolor=colors[i], alpha=alphas[i])
    #     labels.append(chkpt)
    #     i += 1 


    mean_df = fdf.groupby(['ratio']).median()
    std_df = fdf.groupby(['ratio']).std()

    r = 0.15
    mean = mean_df.to_numpy()
    lower_lim = mean - r*std_df.to_numpy()
    upper_lim = np.where(mean + r*std_df.to_numpy() > 0, 0, mean + r*std_df.to_numpy())

    for param in fdf.index.get_level_values(0).unique():
    # for param in ["hand_upper"]:
        if param == "base" or param == "Action parameters":
            continue
        p_df = fdf[param]
        values = p_df.to_numpy()
        indices = p_df.index.get_level_values(0)
        ax.fill_between(indices, values, lower_lim, facecolor=colors[i], alpha=alphas)
        ax.fill_between(indices, values, upper_lim, facecolor=colors[i], alpha=alphas)
        # ax.plot(indices, values, color=colors[i], lw=5)
    ax.plot(indices, mean, linewidth=10, color=linecolors[i], marker='.', ms=40)

    ax.set(xlabel="ratio")
    ax.tick_params(width=3, length=20)
    ax.minorticks_on()
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    # if i == 0:
    #     ax.set(ylabel="\% change in avg_" + metric)
    #     ax.set_ylim(-80, 5)
    ax.set(ylabel="\% change in avg_" + metric)
    ax.set_ylim(-80, 5)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.16, box.width, box.height*0.9])

    # Put a legend to the right of the current axis
    # ax.legend(labels, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.grid(visible=True, which='major', axis='y', linewidth=5, linestyle='-')
    ax.grid(visible=True, which='minor', axis='y', linewidth=3, linestyle='--')

    # figsize: 18.5, 11
    # plt.show()
    save_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "images")
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.join(save_dir, "sensitivity_analysis_region_" + chkpt + "_" + metric + ".png")
    plt.savefig(filename)

    return ax

@click.command()
@click.argument("df_file")
@click.option("--steady-state-cutoff", "-s", help="steps after which model stabilizes", default=6000)
def main(df_file: str, steady_state_cutoff: int):
    df = pd.read_csv(df_file)

    ignore_params = ["rna_alpha", "hand_armature", "hand_effort", "hand_mass", "object_mass", "object_friction", "hand_friction_fingertips", "action_delay_prob", 'affine_action_white', 'affine_action_additive']
    # ignore_params = ['action_delay_prob', 'object_friction', 'hand_friction_fingertips', 'hand_armature', 'affine_action_white', 'affine_action_additive', 'rna_alpha', 'hand_effort']

    for i_param in ignore_params:
        mask = df["param"] == i_param
        df = df[~mask]

    steady_state_df = df[df["step"] > steady_state_cutoff]

    # groups = {"Physical parameters":    ['hand_effort', 'object_mass', 'hand_joint_friction', 'hand_damping', 'hand_stiffness', 'hand_mass', 'object_friction', 'hand_friction_fingertips'],
    #           "Geometric parameters":   ['hand_lower', 'hand_upper', 'hand_armature'],
    #           "Action parameters":      ['affine_action_white', 'affine_action_additive', 'rna_alpha'],
    #           "Observation parameters": ['affine_cube_pose_white', 'affine_dof_pos_additive', 'affine_cube_pose_additive', 'affine_dof_pos_white'],
    #           "Control parameters":     ['action_latency', 'action_delay_prob'],
    #           "base":                   ['base'],}

    # def get_group(param: str):
    #     for group_name, group_list in groups.items():
    #         if param in group_list:
    #             return group_name
    #     return None

    # checkpoints = steady_state_df["checkpoint"].unique()
    # # checkpoints = ["base_chkpt"]
    # for chkpt in checkpoints:
    #     ss_df = steady_state_df[steady_state_df["checkpoint"] == chkpt]

    #     for metric in ["consecutive_successes"]:
    #         metric_ss_df = ss_df[ss_df["metric"] == metric]
    #         group_series = metric_ss_df["param"].apply(get_group)
    #         group_series.name = "group"
    #         metric_ss_df = pd.concat([metric_ss_df, group_series], axis=1, join="inner")
    #         mean_series = metric_ss_df.groupby(['group', 'ratio'])['value'].mean()

    #         temp = [*metric_ss_df["group"].unique()]
    #         temp.remove("base")
    #         multi_index = pd.MultiIndex.from_product([temp, [0.0]], names=['group', 'ratio'])
    #         base_value = mean_series["base", 0.0]
    #         zeros_series = pd.Series(np.ones((len(temp),)) * base_value, index=multi_index)

    #         mean_series = pd.concat([mean_series, zeros_series]).sort_index()

    #         mean_series = (mean_series - mean_series["base"][0.0]) / mean_series["base"][0.0] * 100

    #         print(mean_series.to_string())
            
    #         if chkpt == "base_chkpt":
    #             chkpt = "base_controller"
    #         plot_lines(mean_series, chkpt, metric)


    for metric in ["consecutive_successes"]:
        metric_ss_df = steady_state_df[steady_state_df["metric"] == metric]

        mean_df = metric_ss_df.groupby(['checkpoint', 'param', 'ratio'])['value'].mean().sort_index()
        mean_df.name = "mean"

        # print(mean_df)
        # for chkpt in mean_df.index.get_level_values(0).unique():
        # fig, ax = plt.subplots(1,2, sharey=True, sharex=True)

        # fig.suptitle("Sensitivity Analysis for Robustness")
        # fig.subplots_adjust(top=1.5)
        i = 0
        ax=None
        for chkpt in ["base_chkpt", "ft_delta_ensemble_adv_prob_0.30_value_selection"]:
            chkpt_mean_df: pd.Series = mean_df[chkpt]
            chkpt_mean_df = (chkpt_mean_df - chkpt_mean_df["base"][0.0]) / chkpt_mean_df["base"][0.0] * 100
            temp = [*mean_df.index.get_level_values(1).unique()]
            temp.remove("base")
            multi_index = pd.MultiIndex.from_product([temp, [0.0]], names=['param', 'ratio'])
            base_value = chkpt_mean_df["base", 0.0]
            zeros_series = pd.Series(np.ones((len(temp),)) * base_value, index=multi_index)

            chkpt_mean_df = pd.concat([chkpt_mean_df, zeros_series]).sort_index()
            chkpt_mean_df.name = "mean"
            print(chkpt_mean_df.to_string())
            if chkpt == "base_chkpt":
                chkpt = "base_controller"
            # plot_regions(chkpt_mean_df, chkpt, metric, i)
            i += 1
            plot_lines(chkpt_mean_df, chkpt, metric)

        # save_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "images")
        # os.makedirs(save_dir, exist_ok=True)

        # fig.tight_layout()
        # filename = os.path.join(save_dir, "sensitivity_analysis_region_" + metric + ".png")
        # plt.savefig(filename)

if __name__ == "__main__":
    main()