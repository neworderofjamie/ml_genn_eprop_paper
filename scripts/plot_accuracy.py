import os
import numpy as np
import matplotlib.pyplot as plt
import plot_settings
import seaborn as sns 

from pandas import DataFrame
from glob import glob
from pandas import read_csv


BAR_WIDTH = 1.0
BAR_PAD = 1.1
GROUP_PAD = 2.5

def plot_accuracy_bars(df, axis):
    pal = sns.color_palette()
    bar_x = np.arange(df.shape[0]) * GROUP_PAD
    
    # Show bars for train and test accuracy
    axis.bar(bar_x, df["train_accuracy"] * 100.0, width=BAR_WIDTH, color=pal[0])
    axis.bar(bar_x + BAR_PAD, df["test_accuracy"] * 100.0, width=BAR_WIDTH, color=pal[1])
    
    # Remove axis junk
    sns.despine(ax=axis)
    axis.xaxis.grid(False)
    
    axis.set_xticks(bar_x + (BAR_PAD / 2))
    axis.set_xticklabels(df["config"], rotation=90)
    axis.set_ylabel("Accuracy [%]")
    axis.set_ylim((0, 100.0))
    
# Dictionary to hold data
data = {"config": [], "num_layers": [], "seed": [], 
        "test_accuracy": [], "test_time": [],
        "train_accuracy": [], "train_time": []}

# Loop through test files (signifies complete experiments)
for name in glob(os.path.join("results", "test*.csv")):
    # Split name of test filename into components seperated by _
    name_components = os.path.splitext(os.path.basename(name))[0].split("_")
    
    # **YUCK** dvs-gesture should probably not be _ delimited
    num_components = len(name_components)
    assert ((num_components - 8) % 3) == 0
    
    num_layers = (num_components - 8) // 3
    
    layer_size_component_begin = 8
    layer_recurrent_component_begin = layer_size_component_begin + num_layers
    layer_model_component_begin = layer_recurrent_component_begin + num_layers
    
    config = []
    for i in range(num_layers):
        size = name_components[layer_size_component_begin + i]
        recurrent = (name_components[layer_recurrent_component_begin + i] == "True")
        model = name_components[layer_model_component_begin + i]
        
        config.append(model.upper() + size + ("R" if recurrent else "F"))

    # Read test output CSV
    test_data = read_csv(name, delimiter=",")
    assert test_data.shape[0] == 1
    
    # Read corresponding training output and extract data from last epoch
    train_name = "train_output_" + "_".join(name_components[2:])
    train_data = read_csv(os.path.join("results", train_name) + ".csv", delimiter=",")
    last_epoch_train_data = train_data[train_data["Epoch"] == 99]
    assert last_epoch_train_data.shape[0] == 1

    # Add data to intermediate dictionary
    data["config"].append("-".join(config))
    data["num_layers"].append(num_layers)
    data["seed"].append(int(name_components[7]))
    data["test_accuracy"].append((test_data["Number correct"] / test_data["Num trials"]).iloc[0])
    data["test_time"].append(test_data["Time"].iloc[0])
    data["train_accuracy"].append((last_epoch_train_data["Number correct"] / last_epoch_train_data["Num trials"]).iloc[0])
    data["train_time"].append(last_epoch_train_data["Time"].iloc[0])

# Build dataframe from dictionary and sort by config
df = DataFrame(data=data)
df = df.sort_values("config")

# Split dataframe into one and two layer configurations
one_layer_df = df[df["num_layers"] == 1]
two_layer_df = df[df["num_layers"] == 2]

# Extract best performing one and two layer configurations
best_one_layer = one_layer_df.iloc[df['test_accuracy'].idxmax()]
best_two_layer = two_layer_df.iloc[df['test_accuracy'].idxmax()]
print(f"Best one layer config:{best_one_layer['config']} with {best_one_layer['test_accuracy']}%")
print(f"Best two layer config:{best_two_layer['config']} with {best_two_layer['test_accuracy']}%")


# Create accuracy bar plot
accuracy_fig, accuracy_axes = plt.subplots(1, 2, sharey=True,
                                           figsize=(plot_settings.double_column_width, 2.0))

plot_accuracy_bars(one_layer_df, accuracy_axes[0])
plot_accuracy_bars(two_layer_df, accuracy_axes[1])

accuracy_fig.tight_layout(pad=0)

pareto_fig, pareto_axis = plt.subplots(figsize=(plot_settings.column_width, 2.0))
pareto_axis.scatter(df["test_time"], df["test_accuracy"], s=1)
plt.show()