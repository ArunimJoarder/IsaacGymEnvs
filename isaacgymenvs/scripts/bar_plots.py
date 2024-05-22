import os
import click

import pandas as pd

import matplotlib.pyplot as plt
plt.style.use(os.path.abspath(os.path.join(os.path.dirname(__file__), "presentation.mplstyle")))
from matplotlib.ticker import MultipleLocator

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import numpy as np

BARWIDTH = 0.5

def plot_err_bars(info_df: pd.DataFrame, metric: str):
    fig, ax = plt.subplots()

    order = {0: "Base Controller",
             1: "Noise Generator 0",
             2: "Noise Generator 1",
             3: "Noise Generator 2",
             4: "Noise Generator 3",
             5: "Noise Generator 4",
             }

    colors = ["tab:brown",
              "tab:blue",
              "tab:orange",
              "tab:green",
              "tab:red",
              "tab:purple",
              ]

    ax.minorticks_on()
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.set_axisbelow(True)
    ax.grid(visible=True, which='major', axis='y', linewidth=5, linestyle='-')
    ax.grid(visible=True, which='minor', axis='y', linewidth=3, linestyle='--')

    num_bars = len(info_df.index.unique())
    xticks = 0.75*np.arange(num_bars)
    # i = 0
    for i, ind in order.items():
        # row = info_df[ind]
        # print(row)
        mean = info_df["mean"][ind]
        std  = info_df["std"][ind]

        ax.bar(xticks[i], mean, width = BARWIDTH, label=ind, facecolor=colors[i])
        ax.errorbar(xticks[i], mean, yerr=std, fmt='o', color='k', capsize=15, capthick=4)

        # print(i, mean, std)
        # i += 1
    # ax.bar(xticks[0], info_df["mean"]["Base Controller"], width=BARWIDTH, label="Base Controller")
    # ax.errorbar(xticks[0], info_df["mean"]["base_controller"], yerr=info_df["std"]["base_controller"], fmt='o', color='k', capsize=15, capthick=4)

    ax.set_title("Controller performance with different adversaries")
    ax.set_xticks(xticks)
    ax.set_xticklabels(['', '', '', '', '', '']) 
    ax.set(ylabel="avg_" + metric)
    ax.tick_params(width=3, length=20)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.16, box.width, box.height*0.9])

    # Put a legend to the right of the current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.48, -0.05), ncol=3)

    # figsize: 18.5, 11
    # plt.show()
    save_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "images")
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.join(save_dir, "bar_plot_" + metric + ".png")
    # filename = os.path.join(save_dir, "sensitivity_analysis_" + chkpt + "_" + metric + "_presentation.png")
    plt.savefig(filename)

@click.command()
@click.argument("df_file")
@click.option("--steady-state-cutoff", "-s", help="steps after which model stabilizes", default=6000)
def main(df_file: str, steady_state_cutoff: int):
    df = pd.read_csv(df_file)

    ########## #TODO: add metrics ###########
    metric_list = ["consecutive_successes"]

    df = df.loc[df["metric"].isin(metric_list)]
    ss_df = df[df["step"] > steady_state_cutoff]

    print(ss_df)

    ########## #TODO: change labels ###########
    def update_labels(label: str):
        if label == "base":
            label = "Base Controller"
        elif "adv_obs_acts_ensemble_kl_loss_train_upd_2_kl_loss_coeff" in label:
            label = "Noise Generator " + label[-1]

        # if label == "mcs_500_last_euler_ft_delta_ensemble_adv_prob_0.30_no_value_selection_adr_ranges_8192_6000_iters_ep_6000_rew__3517.71_":
        #     label = "Residual Network + ensemble adversaries (random selection)"
        # elif label == "mcs_500_last_euler_ft_delta_ensemble_adv_prob_0.30_value_selection_adr_ranges_8192_6000_iters_ep_6000_rew__3505.26_":
        #     label = "Residual Network + ensemble adversaries (max. value selection)"
        # elif label == "mcs_500_last_euler_ft_delta_no_ensemble_adv_prob_0.30_adr_ranges_8192_6000_iters_ep_6000_rew__3569.61_":
        #     label = "Residual Network + single adversary"
        # elif label == "mcs_500_last_euler_ft_delta_no_ensemble_no_adv_adr_ranges_8192_6000_iters_ep_6000_rew__3308.98_":
        #     label = "Residual Network (no adversary)"
        # elif label == "mcs_500_base":
        #     label = "Base Controller"
        return label


    for metric in ss_df["metric"].unique():
        metric_df = ss_df[ss_df["metric"] == metric]

        metric_df["experiment"] = metric_df["experiment"].apply(update_labels)

        mean_df = metric_df.groupby("experiment")["value"].mean()
        mean_df.name = "mean"

        std_df = metric_df.groupby("experiment")["value"].std()
        std_df.name = "std"
        
        info_df = pd.concat([mean_df, std_df], axis=1, join='inner')

        ########## #TODO: remove data ###########
        drop_list = ["euler_ensemble_adv_obs_acts_max_value_selection_base",
                     "euler_ensemble_adv_obs_acts_no_value_selection_base"]
        # drop_list = ["mcs_500_last_euler_ft_ensemble_adv_prob_0.30_no_value_selection_adr_ranges_4096_104000_iters_ep_104000_rew__3830.88_fixed",
        #              "mcs_500_last_euler_ft_ensemble_adv_prob_0.30_value_selection_adr_ranges_4096_104000_iters_ep_104000_rew__3628.27_fixed",
        #              "mcs_500_last_euler_ft_no_ensemble_adv_prob_0.30_adr_ranges_4096_104000_iters_ep_104000_rew__4356.1_fixed",
        #              "mcs_500_last_euler_ft_no_ensemble_no_adv_adr_ranges_4096_104000_iters_ep_104000_rew__4035.65_fixed"]
        info_df = info_df.drop(drop_list, axis=0)
        
        print(info_df.to_string())
        plot_err_bars(info_df, metric)

if __name__ == "__main__":
    main()