import os
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plot_settings
import seaborn as sns 

from pandas import DataFrame, NamedAgg

from glob import glob
from itertools import product
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import concat, read_csv


BAR_WIDTH = 1.0
BAR_PAD = 1.1
GROUP_PAD = 2.5

def plot_accuracy_bars(df, axis):
    pal = sns.color_palette()
    bar_x = np.arange(df.shape[0]) * GROUP_PAD
    
    # Show bars for train and test accuracy
    train = axis.bar(bar_x, df["mean_train_accuracy"], yerr=df["sd_train_accuracy"],
             width=BAR_WIDTH, color=pal[0])
    test = axis.bar(bar_x + BAR_PAD, df["mean_test_accuracy"], yerr=df["sd_test_accuracy"],
             width=BAR_WIDTH, color=pal[1])
    
    # Remove axis junk
    sns.despine(ax=axis)
    axis.xaxis.grid(False)
    
    axis.set_xticks(bar_x + (BAR_PAD / 2))
    return train, test

def plot_accuracy_heatmap(df, width, height, cmap_size, *sparsity_series):
    # Loop through all two layer sparsity configurations
    lookup = {10: 2, 5: 1, 1: 0}
    test_heat = np.zeros([len(lookup)] * len(sparsity_series))
    train_heat = np.zeros([len(lookup)] * len(sparsity_series))
    for i in sparsity_series[0].index:
        # Use lookup dictionary to calculate index
        # **THINK** list of tuples!
        index = tuple(lookup[int(s.loc[i])] for s in sparsity_series)

        # Copy mean test and train accuracies into heatamp
        test_heat[index] = df.loc[i]["mean_test_accuracy"]
        train_heat[index] = df.loc[i]["mean_train_accuracy"]

    # Create two column figure
    fig, train_axis = plt.subplots(figsize=(width, height))

    # **YUCK** the only way I can figure out to make the colorbar the same height as the imshows is to use an AxesDivider. However
    # using this with a figure created with two panels and sharey breaks the y-axis. Instead we need to use the axes divider to create all axes
    divider = make_axes_locatable(train_axis)

    # Split off testing axis
    test_axis = divider.append_axes("right", size="100%")

    # Split of smaller colorbar axis
    colorbar_axis = divider.append_axes("right", size=cmap_size, pad=0.05)

    # Plot train and test performance heatmaps
    pcolor_kwargs = {"vmin": 70, "vmax": 100, "cmap": "Reds",
                     "edgecolors": "white", "linewidths": 0.5, "linestyle": "--"}#, "norm": colors.PowerNorm(1.5)}
    
    # Reshape heatmaps to 2D
    if len(train_heat.shape) > 2:
        train_heat = np.reshape(train_heat, (len(lookup), -1))
    if len(test_heat.shape) > 2:
        test_heat = np.reshape(test_heat, (len(lookup), -1))

    im = train_axis.pcolor(train_heat, **pcolor_kwargs)
    test_axis.pcolor(test_heat, **pcolor_kwargs)

    # Find index of best test performance
    best_test = np.unravel_index(np.argmax(test_heat), test_heat.shape)
    
    # Create a Rectangle patch
    rect = patches.Rectangle((best_test[1], best_test[0]), 1.0, 1.0, 
                             linewidth=1, edgecolor="white", facecolor="none")
    # Add the patch to the Axes
    test_axis.add_patch(rect)
    
    # Add text showing accuracy
    test_axis.text(best_test[1] + 0.5, best_test[0] + 0.5,
                   f"{test_heat[best_test]:.2f}%",
                   color="white", ha="center", va="center", size=6 if plot_settings.paper else "small")
    
    
    # Add color bar
    fig.colorbar(im, cax=colorbar_axis, orientation="vertical")

    train_axis.set_title("A", loc="left")
    test_axis.set_title("B", loc="left")
    test_axis.set_aspect("equal")
    train_axis.set_aspect("equal")
    # Loop through image axes
    for a in [train_axis, test_axis]:
        sns.despine(ax=a)
        a.xaxis.grid(False)
        a.yaxis.grid(False)
    
    return fig, train_axis, test_axis

# Dictionary to hold data
data = {"config": [], "sparse": [], "sparse_config": [], "num_layers": [], "seed": [], 
        "test_accuracy": [], "test_time": [],
        "train_accuracy": [], "train_time": []}

# Loop through test files (signifies complete experiments)
for name in glob(os.path.join("results", "test_output_*.csv")):
    # Split name of test filename into components seperated by _
    name_components = os.path.splitext(os.path.basename(name))[0].split("_")
    
    # **YUCK** dvs-gesture should probably not be _ delimited - stops this generalising
    num_components = len(name_components)
    assert ((num_components - 8) % 5) == 0
    
    num_layers = (num_components - 8) // 5
    
    layer_size_component_begin = 8
    layer_recurrent_component_begin = layer_size_component_begin + num_layers
    layer_model_component_begin = layer_recurrent_component_begin + num_layers
    layer_input_sparsity_component_begin = layer_model_component_begin + num_layers
    layer_recurrent_sparsity_component_begin = layer_input_sparsity_component_begin + num_layers
    config = []
    sparse_config = []
    sparse = False
    skip = False
    for i in range(num_layers):
        size = name_components[layer_size_component_begin + i]
        recurrent = (name_components[layer_recurrent_component_begin + i] == "True")
        model = name_components[layer_model_component_begin + i]
        input_sparsity = name_components[layer_input_sparsity_component_begin + i]
        recurrent_sparsity = name_components[layer_recurrent_sparsity_component_begin + i]
        
        if input_sparsity != "1.0" or recurrent_sparsity != "1.0":
            sparse = True
        
        # **YUCK** not really happy with larger configurations so skip
        if int(size) > 512:
            skip = True

        config.append(model.upper() + size + ("R" if recurrent else "F"))
        sparse_config.append(f"I:{int(100.0 * float(input_sparsity))}%" + (f",R:{int(100.0 * float(recurrent_sparsity))}%" if recurrent else ""))

    # Read test output CSV
    test_data = read_csv(name, delimiter=",")
    
    if "Epoch" in test_data:
        last_epoch_test_data = test_data[test_data["Epoch"] == 99]
    else:
        last_epoch_test_data = test_data
    assert last_epoch_test_data.shape[0] == 1

    # Read corresponding training output and extract data from last epoch
    train_name = "train_output_" + "_".join(name_components[2:])
    train_data = read_csv(os.path.join("results", train_name) + ".csv", delimiter=",")
    last_epoch_train_data = train_data[train_data["Epoch"] == 99]
    assert last_epoch_train_data.shape[0] == 1

    # Add data to intermediate dictionary
    if not skip:
        data["config"].append("-".join(config))
        data["sparse"].append(sparse)
        data["sparse_config"].append("-".join(sparse_config))
        data["num_layers"].append(num_layers)
        data["seed"].append(int(name_components[7]))
        data["test_accuracy"].append((100.0 * (last_epoch_test_data["Number correct"] / last_epoch_test_data["Num trials"])).iloc[0])
        data["test_time"].append(last_epoch_test_data["Time"].iloc[0])
        data["train_accuracy"].append((100.0 * (last_epoch_train_data["Number correct"] / last_epoch_train_data["Num trials"])).iloc[0])
        data["train_time"].append(last_epoch_train_data["Time"].iloc[0])

# Build dataframe from dictionary and sort by config
df = DataFrame(data=data)
df = df.sort_values("config")

# Group data by config and number of layers (later just to ensure column is retained)
df = df.groupby(["config", "num_layers", "sparse", "sparse_config"], as_index=False)
df = df.agg(mean_test_accuracy=NamedAgg(column="test_accuracy", aggfunc=np.mean),
            sd_test_accuracy=NamedAgg(column="test_accuracy", aggfunc=np.std),
            mean_test_time=NamedAgg(column="test_time", aggfunc=np.mean),
            sd_test_time=NamedAgg(column="test_time", aggfunc=np.std),
            mean_train_accuracy=NamedAgg(column="train_accuracy", aggfunc=np.mean),
            sd_train_accuracy=NamedAgg(column="train_accuracy", aggfunc=np.std),
            mean_train_time=NamedAgg(column="train_time", aggfunc=np.mean),
            sd_train_time=NamedAgg(column="train_time", aggfunc=np.std))

print(f"{len(df[df['sparse'] == False])} dense configurations")
print(f"{len(df[df['sparse'] == True])} sparse configurations")

# Split dataframe into one and two layer configurations for dense and sparse
one_layer_dense_df = df[(df["num_layers"] == 1) & (df["sparse"] == False)]
two_layer_dense_df = df[(df["num_layers"] == 2) & (df["sparse"] == False)]
one_layer_sparse_df = df[(df["num_layers"] == 1) & (df["sparse"] == True)]
two_layer_sparse_df = df[(df["num_layers"] == 2) & (df["sparse"] == True)]

# Extract best performing configurations
best_one_layer_dense = one_layer_dense_df.loc[one_layer_dense_df['mean_test_accuracy'].idxmax()]
best_two_layer_dense = two_layer_dense_df.loc[two_layer_dense_df['mean_test_accuracy'].idxmax()]
best_one_layer_sparse = one_layer_sparse_df.loc[one_layer_sparse_df['mean_test_accuracy'].idxmax()]
best_two_layer_sparse = two_layer_sparse_df.loc[two_layer_sparse_df['mean_test_accuracy'].idxmax()]
print(f"Best one layer dense config:{best_one_layer_dense['config']} with {best_one_layer_dense['mean_test_accuracy']:.2f}±{best_one_layer_dense['sd_test_accuracy']:.2f}%")
print(f"Best two layer dense config:{best_two_layer_dense['config']} with {best_two_layer_dense['mean_test_accuracy']:.2f}±{best_two_layer_dense['sd_test_accuracy']:.2f}%")
print(f"Best one layer sparse config:{best_one_layer_sparse['config']} {best_one_layer_sparse['sparse_config']} with {best_one_layer_sparse['mean_test_accuracy']:.2f}±{best_one_layer_sparse['sd_test_accuracy']:.2f}%")
print(f"Best two layer sparse config:{best_two_layer_sparse['config']} {best_two_layer_sparse['sparse_config']} with {best_two_layer_sparse['mean_test_accuracy']:.2f}±{best_two_layer_sparse['sd_test_accuracy']:.2f}%")

# Create dense accuracy bar plot
dense_fig, dense_axes = plt.subplots(1, 2, sharey=True,
                                     figsize=(plot_settings.double_column_width, 2.0))

train_actor, test_actor = plot_accuracy_bars(one_layer_dense_df, dense_axes[0])
plot_accuracy_bars(two_layer_dense_df, dense_axes[1])
dense_axes[0].set_xticklabels(one_layer_dense_df["config"], rotation=90)
dense_axes[1].set_xticklabels(two_layer_dense_df["config"], rotation=90)
dense_axes[0].set_title("A", loc="left")
dense_axes[1].set_title("B", loc="left")
dense_axes[0].set_ylabel("Accuracy [%]")
dense_axes[0].set_ylim((80.0, 100.0))
dense_fig.legend([train_actor, test_actor], ["Train", "Test"], 
                 ncol=2, loc="lower left", frameon=False)
dense_fig.tight_layout(pad=0)

# **YUCK** split one layer sparse config strings back into seperate strings
one_layer_sparsity = one_layer_sparse_df["sparse_config"].str.split(",", expand=True)

# Remove "I:" and "R:" prefixes and "%" suffixes
one_layer_pop0_pop1_sparsity = one_layer_sparsity[0].str.slice(2, -1)
one_layer_pop1_pop1_sparsity = one_layer_sparsity[1].str.slice(2, -1)

# Plot heatmap
one_layer_sparse_fig, one_layer_sparse_train_axis, one_layer_sparse_test_axis =\
    plot_accuracy_heatmap(one_layer_sparse_df, plot_settings.column_width, 1.6, "15%",
                          one_layer_pop0_pop1_sparsity, one_layer_pop1_pop1_sparsity)

one_layer_sparse_train_axis.set_ylabel("Input connectivity")

# Set y tick labels
sparsities = ["1%", "5%", "10%"]
one_layer_sparse_train_axis.set_yticks(np.linspace(0.5, 2.5, 3))
one_layer_sparse_train_axis.set_yticklabels(sparsities)
one_layer_sparse_test_axis.set_yticks([])

for a in [one_layer_sparse_train_axis, one_layer_sparse_test_axis]:
    a.set_xlabel("Recurrent connectivity")
    
    # Set x tick labels
    a.set_xticks(np.linspace(0.5, 2.5, 3))
    a.set_xticklabels(sparsities)

one_layer_sparse_fig.tight_layout(pad=0.3)

# **YUCK** split two layer sparse config strings back into seperate strings
two_layer_sparse_config_split = two_layer_sparse_df["sparse_config"].str.split("-", expand=True)
two_layer_recurrent_sparsity = two_layer_sparse_config_split[1].str.split(",", expand=True)

# Remove "I:" and "R:" prefixes and "%" suffixes
two_layer_pop0_pop1_sparsity = two_layer_sparse_config_split[0].str.slice(2, -1)
two_layer_pop1_pop2_sparsity = two_layer_recurrent_sparsity[0].str.slice(2, -1)
two_layer_pop2_pop2_sparsity = two_layer_recurrent_sparsity[1].str.slice(2, -1)

# Plot heatmap
two_layer_sparse_fig, two_layer_sparse_train_axis, two_layer_sparse_test_axis =\
    plot_accuracy_heatmap(two_layer_sparse_df, plot_settings.double_column_width, 1.6, "5%",
                          two_layer_pop0_pop1_sparsity, two_layer_pop1_pop2_sparsity, 
                          two_layer_pop2_pop2_sparsity)

# Set y tick labels
two_layer_sparse_train_axis.set_ylabel("Input connectivity")
two_layer_sparse_train_axis.set_yticks(np.linspace(0.5, 2.5, 3))
two_layer_sparse_train_axis.set_yticklabels(sparsities)
two_layer_sparse_test_axis.set_yticks([])

# Loop through image axes
for a in [two_layer_sparse_train_axis, two_layer_sparse_test_axis]:
    a.set_xlabel("Hidden connectivity")
    
    # Set x tick labels
    a.set_xticks(np.linspace(0.5, 8.5, 9))
    a.set_xticklabels([f"I:{i}\nR:{j}" for i, j in product(sparsities, repeat=2)])

two_layer_sparse_fig.tight_layout(pad=1.4, rect=[0.0, 0.0,
                                                 1.0, 0.8])

best_accuracy_fig, best_accuracy_axis = plt.subplots(figsize=(plot_settings.column_width, 2.0))

# Concatenate together best configurations and transpose to return to original format
best_data = concat([best_one_layer_dense, best_two_layer_dense,
                    best_one_layer_sparse, best_two_layer_sparse], axis=1)
best_data = best_data.transpose()
tick_labels = [f"{conf}\n{sp_conf}" if sp else f"{conf}\nDense"
               for conf, sp_conf, sp in zip(best_data["config"],
                                            best_data["sparse_config"],
                                            best_data["sparse"])]
train_actor, test_actor = plot_accuracy_bars(best_data, best_accuracy_axis)
best_accuracy_axis.set_xticklabels(tick_labels, size=(5.5 if plot_settings.paper else 15.0))
best_accuracy_axis.set_ylabel("Accuracy [%]")
best_accuracy_axis.set_ylim((80.0, 100.0))
best_accuracy_fig.legend([train_actor, test_actor], ["Train", "Test"], 
                         ncol=2, loc="lower center", frameon=False)
best_accuracy_fig.tight_layout(pad=0, rect=[0.0, (0.125 if plot_settings.paper else 0.0),
                                            1.0, 1.0])

if not plot_settings.presentation and not plot_settings.poster:
    dense_fig.savefig("../figures/dense_accuracy.pdf")
    one_layer_sparse_fig.savefig("../figures/sparse_accuracy.pdf")
    two_layer_sparse_fig.savefig("../figures/two_layer_sparse_accuracy.pdf")
    best_accuracy_fig.savefig("../figures/best_accuracy.pdf")
plt.show()
