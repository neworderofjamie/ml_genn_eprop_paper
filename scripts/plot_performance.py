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
#test_data = test_data.sort_values("Config")
#train_data = train_data.sort_values("Config")

# Calculate timesteps/second
test_fps = (TEST_EXAMPLES * TEST_STEPS) / test_data["Inference time [s]"]
train_fps = (TRAIN_EXAMPLES * TRAIN_STEPS) / train_data["Epoch train time [s]"]

gpu_test = test_data[test_data["Backend"] == "GPU"]
cpu_test = test_data[test_data["Backend"] == "CPU"]

gpu_test_fps = test_fps.loc[gpu_test.index].values
cpu_test_fps = test_fps.loc[cpu_test.index].values

fptt_fps = TRAIN_EXAMPLES / 0.4
print(f"Training speedup vs FPTT {train_fps.values[-1] / fptt_fps}")
print(f"Sparse ALIF512R CPU inference speedup {cpu_test_fps[2] / cpu_test_fps[0]}")
print(f"Sparse ALIF256F256 CPU inference speedup {cpu_test_fps[3] / cpu_test_fps[1]}")
print(f"Sparse ALIF512R training speedup {train_fps.values[2] / train_fps.values[0]}")
print(f"Sparse ALIF256F256 training speedup {train_fps.values[3] / train_fps.values[1]}")

# Get configurations
# **HACK** cos I cba to write a sorting function
#cconfigurations = np.intersect1d(test_data["Config"].unique(),
#                                train_data["Config"].unique())
configurations = ["512R", "256F256R", "512R sparse",
                  "256F256R sparse", "1024F512R"]

fig, axis = plt.subplots(figsize=(plot_settings.column_width, 1.7))

pal = sns.color_palette()
bar_x = np.arange(len(configurations)) * GROUP_PAD

# Show bars for train and test accuracy
train = axis.bar(bar_x[:len(train_fps.values)], train_fps.values,
         width=BAR_WIDTH, color=pal[0])
inference_gpu = axis.bar(bar_x[:len(gpu_test_fps)] + BAR_PAD, gpu_test_fps,
         width=BAR_WIDTH, color=pal[1])
inference_cpu = axis.bar(bar_x[:len(cpu_test_fps)] + (2 * BAR_PAD), cpu_test_fps,
         width=BAR_WIDTH, color=pal[2])
fptt = axis.bar([bar_x[-1] + BAR_PAD], [fptt_fps], 
                width=BAR_WIDTH, color=pal[3])
axis.axhline(1000.0, linestyle="--", color="gray")

# Remove axis junk
sns.despine(ax=axis)
axis.xaxis.grid(False)
axis.set_yscale("log")
axis.set_ylabel("Timesteps per second", size=("small" if plot_settings.poster else None))
axis.set_xticks(bar_x + BAR_PAD)
axis.set_xticklabels([c.replace(" ", "\n") for c in configurations])

fig.legend([train, fptt, inference_gpu, inference_cpu],
           ["GeNN Train", "FPTT train", "GeNN GPU inference", "GeNN CPU inference"],
           loc="lower center", ncol=2, frameon=False)
fig.tight_layout(pad=0, rect=[0.0, (0.0 if plot_settings.poster else 0.225), 1.0, 1.0])

if not plot_settings.presentation and not plot_settings.poster:
    fig.savefig("../figures/performance.pdf")

plt.show()
