import os
import numpy as np
import matplotlib.pyplot as plt
import plot_settings
import seaborn as sns 

from glob import glob
from pandas import read_csv

BAR_WIDTH = 1.0
BAR_PAD = 1.1
GROUP_PAD = 4.0

TRAIN_STEPS = 18457
TEST_STEPS = 15592

TRAIN_EXAMPLES = 1077
TEST_EXAMPLES = 264

# Load data
test_data = read_csv("results/test_performance.csv", delimiter=",")
train_data = read_csv("results/train_performance.csv", delimiter=",")
test_data = test_data.sort_values("Config")
train_data = train_data.sort_values("Config")

# Calculate timesteps/second
test_fps = (TEST_EXAMPLES * TEST_STEPS) / test_data["Inference time [s]"]
train_fps = (TRAIN_EXAMPLES * TRAIN_STEPS) / train_data["Epoch train time [s]"]

gpu_test = test_data[test_data["Backend"] == "GPU"]
cpu_test = test_data[test_data["Backend"] == "CPU"]

# Get configurations
configurations = np.intersect1d(test_data["Config"].unique(),
                                train_data["Config"].unique())

fig, axis = plt.subplots(figsize=(plot_settings.column_width, 2.0))

pal = sns.color_palette()
bar_x = np.arange(len(configurations)) * GROUP_PAD

# Show bars for train and test accuracy
train = axis.bar(bar_x, train_fps.values,
         width=BAR_WIDTH, color=pal[0])
inference_gpu = axis.bar(bar_x + BAR_PAD, test_fps.loc[gpu_test.index].values,
         width=BAR_WIDTH, color=pal[1])
inference_cpu = axis.bar(bar_x + (2 * BAR_PAD), test_fps.loc[cpu_test.index].values,
         width=BAR_WIDTH, color=pal[2])

axis.axhline(1000.0, linestyle="--", color="gray")

# Remove axis junk
sns.despine(ax=axis)
axis.xaxis.grid(False)
axis.set_yscale("log")
axis.set_ylabel("Timesteps per second")
axis.set_xticks(bar_x + BAR_PAD)
axis.set_xticklabels(configurations)

fig.legend([train, inference_gpu, inference_cpu],
           ["Train", "GPU inference", "CPU inference"],
           loc="lower center", ncol=3)
fig.tight_layout(pad=0, rect=[0.0, 0.125, 1.0, 1.0])

if not plot_settings.presentation and not plot_settings.poster:
    fig.savefig("../figures/performance.pdf")

plt.show()