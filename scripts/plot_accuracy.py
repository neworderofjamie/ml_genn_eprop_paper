import os
import numpy as np
import matplotlib.pyplot as plt
import plot_settings
import seaborn as sns 

from pandas import DataFrame, NamedAgg
from glob import glob
from pandas import read_csv


BAR_WIDTH = 1.0
BAR_PAD = 1.1
GROUP_PAD = 2.5

def plot_accuracy_bars(df, axis):
    pal = sns.color_palette()
    bar_x = np.arange(df.shape[0]) * GROUP_PAD
    
    # Show bars for train and test accuracy
    axis.bar(bar_x, df["mean_train_accuracy"], yerr=df["sd_train_accuracy"],
             width=BAR_WIDTH, color=pal[0])
    axis.bar(bar_x + BAR_PAD, df["mean_test_accuracy"], yerr=df["sd_test_accuracy"],
             width=BAR_WIDTH, color=pal[1])
    
    # Remove axis junk
    sns.despine(ax=axis)
    axis.xaxis.grid(False)
    
    axis.set_xticks(bar_x + (BAR_PAD / 2))

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
dense_accuracy_fig, dense_accuracy_axes = plt.subplots(1, 2, sharey=True,
                                                       figsize=(plot_settings.double_column_width, 2.0))

plot_accuracy_bars(one_layer_dense_df, dense_accuracy_axes[0])
plot_accuracy_bars(two_layer_dense_df, dense_accuracy_axes[1])
dense_accuracy_axes[0].set_xticklabels(one_layer_dense_df["config"], rotation=90)
dense_accuracy_axes[1].set_xticklabels(two_layer_dense_df["config"], rotation=90)
dense_accuracy_axes[0].set_title("A", loc="left")
dense_accuracy_axes[1].set_title("B", loc="left")
dense_accuracy_axes[0].set_ylabel("Accuracy [%]")
dense_accuracy_axes[0].set_ylim((80.0, 100.0))
dense_accuracy_fig.tight_layout(pad=0)

# Create sparse accuracy bar plot
one_layer_sparse_accuracy_fig, one_layer_sparse_accuracy_axis = plt.subplots(figsize=(plot_settings.column_width, 2.0))

plot_accuracy_bars(one_layer_sparse_df, one_layer_sparse_accuracy_axis)
one_layer_sparse_accuracy_axis.set_ylabel("Accuracy [%]")
one_layer_sparse_accuracy_axis.set_ylim((80.0, 100.0))
one_layer_sparse_accuracy_axis.set_xticklabels([c.replace(",", "\n") for c in one_layer_sparse_df["sparse_config"]])
one_layer_sparse_accuracy_fig.tight_layout(pad=0)



# **YUCK** split two layer sparse config strings back into seperate strings
two_layer_sparse_config_split = two_layer_sparse_df["sparse_config"].str.split("-", expand=True)
two_layer_recurrent_sparsity = two_layer_sparse_config_split[1].str.split(",", expand=True)

# Remove "I:" and "R:" prefixes and "%" suffixes
two_layer_pop0_pop1_sparsity = two_layer_sparse_config_split[0].str.slice(2, -1)
two_layer_pop1_pop2_sparsity = two_layer_recurrent_sparsity[0].str.slice(2, -1)
two_layer_pop2_pop2_sparsity = two_layer_recurrent_sparsity[1].str.slice(2, -1)

# Build heatmaps of test and train accuracy
lookup = {10: 0, 5: 1, 1: 2}
test_heat = np.zeros((3, 3, 3))
train_heat = np.zeros((3, 3, 3))
for i in two_layer_pop0_pop1_sparsity.index:
    index = np.index_exp[lookup[int(two_layer_pop0_pop1_sparsity.loc[i])],
                         lookup[int(two_layer_pop1_pop2_sparsity.loc[i])],
                         lookup[int(two_layer_pop2_pop2_sparsity.loc[i])]]
    test_heat[index] = two_layer_sparse_df.loc[i]["mean_test_accuracy"]
    train_heat[index] = two_layer_sparse_df.loc[i]["mean_train_accuracy"]

one_layer_sparse_accuracy_fig, one_layer_sparse_accuracy_axes =\
    plt.subplots(1, 2, sharey=True, figsize=(plot_settings.double_column_width, 2.0))

one_layer_sparse_accuracy_axes[0].imshow(train_heat.reshape((3, 3 * 3)))
one_layer_sparse_accuracy_axes[1].imshow(test_heat.reshape((3, 3 * 3)))
if not plot_settings.presentation and not plot_settings.poster:
    dense_accuracy_fig.savefig("../figures/dense_accuracy.pdf")
    one_layer_sparse_accuracy_fig.savefig("../figures/sparse_accuracy.pdf")

plt.show()
