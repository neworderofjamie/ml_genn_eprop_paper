import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_settings
import seaborn as sns 

from glob import glob

configs = ["256_256_False_True_alif_alif_1.0_1.0_1.0_1.0",
           "512_True_alif_1.0_1.0",
           "512_True_alif_0.05_0.1"]

fig, axis = plt.subplots(figsize=(plot_settings.column_width, 2.0))

# Loop through configurations
actors = []
for c in configs:
    # Read data and concatenate all repeates
    df = pd.concat([pd.read_csv(f, delimiter=",") 
                    for f in glob(f"results/test_output_0_512_100_dvs_gesture_*_{c}.csv")])
    
    # Group data by epoch index
    df = df.groupby(["Epoch"], as_index=False)

    # Aggregate to calculate mean and std
    df = df.agg(mean_correct=pd.NamedAgg(column="Number correct", aggfunc=np.mean),
               std_correct=pd.NamedAgg(column="Number correct", aggfunc=np.std))

    
    actors.append(axis.plot(df["Epoch"], 100.0 * (df["mean_correct"] / 264))[0])
    axis.fill_between(df["Epoch"], 100.0 * ((df["mean_correct"] - df["std_correct"]) / 264),
                      100.0 * ((df["mean_correct"] + df["std_correct"]) / 264), alpha=0.2)

axis.set_xlabel("Epoch")
axis.set_ylabel("Accuracy [%]")
sns.despine(ax=axis)
axis.xaxis.grid(False)
fig.legend(actors, ["ALIF256F-ALIF256R", "ALIF512R", "ALIF512R sparse"],
           loc="lower center", ncol=3)
fig.tight_layout(pad=0, rect=[0.0, 0.125, 1.0, 1.0])

if not plot_settings.presentation and not plot_settings.poster:
    fig.savefig("../figures/learning.pdf")

plt.show()