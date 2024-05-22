import os
import click

import pandas as pd

import matplotlib.pyplot as plt
plt.style.use(os.path.abspath(os.path.join(os.path.dirname(__file__), "presentation.mplstyle")))

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import numpy as np

def plot_clusters(clusters, names):
    fig, ax = plt.subplots()

    names = ["Noise Generator 0",
             "Noise Generator 1",
             "Noise Generator 2",
             "Noise Generator 3",
             "Noise Generator 4",
             ]

    ax.set_title("PCA of noise generated by ensemble noise generators")

    ax.set(ylabel="Principle Component 2", xlabel="Principle Component 1")
    ax.tick_params(width=3, length=20)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.16, box.width, box.height*0.9])

    for cluster in clusters:
        ax.scatter(cluster["PC1"], cluster["PC2"], s=500)

    # Put a legend to the right of the current axis
    ax.legend(names, loc='upper center', bbox_to_anchor=(0.5, -0.125), ncol=3)
    # plt.grid(visible=True, which='major', axis='y', linewidth=5, linestyle='-')

    # plt.show()
    save_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "images", "ensemble")
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.join(save_dir, "ensemble_clusters.png")
    plt.savefig(filename)

@click.command()
@click.argument("df_file")
@click.option("--steady-state-cutoff", "-s", help="steps after which model stabilizes", default=6000)
def main(df_file: str, steady_state_cutoff: int):
    df = pd.read_csv(df_file)

    noise_list = ["action_noise_avg", "dof_pos_noise_avg", "object_pose_noise_avg/pos", "object_pose_noise_avg/rot"]
    df = df.loc[df["metric"].isin(noise_list)]

    steady_state_df = df[df["step"] > steady_state_cutoff]

    print(steady_state_df.groupby(["experiment", "metric"])["value"].mean().to_string())

    temp_df = steady_state_df.pivot(index=['experiment', 'step'], columns='metric', values='value').reset_index()

    temp_df = temp_df.drop(["step"], axis=1)
    # print(temp_df)

    numer_df = temp_df[noise_list]
    scaler = StandardScaler()
    numer_df = pd.DataFrame(scaler.fit_transform(numer_df))
    numer_df.columns = noise_list
    numer_df = pd.concat([temp_df["experiment"], numer_df], axis=1, join="inner")
    # print(numer_df)


    plotX = pd.DataFrame(np.array(numer_df))
    plotX.columns = numer_df.columns

    pca_2d = PCA(n_components=2)

    PCs_2d = pd.DataFrame(pca_2d.fit_transform(plotX.drop('experiment', axis=1)))
    PCs_2d.columns = ["PC1", "PC2"]

    plotX = pd.concat([plotX, PCs_2d], axis=1, join="inner")


    cluster_names = plotX["experiment"].unique()
    # print(cluster_names)

    clusters = []
    for name in cluster_names:
        a = plotX[plotX["experiment"] == name]
        clusters.append(a)
        # print(name, a)
    # print(clusters)
    # print(numer_df.head())
    # print(temp_df.head())
    # plotX.to_csv(df_file[:-4] + "_pivoted.csv")

    plot_clusters(clusters, cluster_names)

if __name__ == "__main__":
    main()