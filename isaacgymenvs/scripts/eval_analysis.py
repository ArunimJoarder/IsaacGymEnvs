import os
import click

import pandas as pd

import matplotlib.pyplot as plt
plt.style.use(os.path.abspath(os.path.join(os.path.dirname(__file__), "presentation.mplstyle")))

import matplotlib.cm as mplcm
import matplotlib.colors as colors
import numpy as np

@click.command()
@click.argument("df_file")
@click.option("--steady-state-cutoff", "-s", help="steps after which model stabilizes", default=6000)
def main(df_file: str, steady_state_cutoff: int):
    df = pd.read_csv(df_file)

    steady_state_df = df[df["step"] > steady_state_cutoff]
    
    mean_df = steady_state_df.groupby(['experiment', 'metric'])['value'].mean().sort_index().rename("mean")
    std_df = steady_state_df.groupby(['experiment', 'metric'])['value'].std().sort_index().rename("std")

    eval_df = pd.concat([mean_df, std_df], axis=1)

    print(eval_df.to_string())

if __name__ == "__main__":
    main()