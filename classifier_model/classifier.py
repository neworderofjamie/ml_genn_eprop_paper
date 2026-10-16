import csv
import os
import numpy as np

from argparse import ArgumentParser
from ml_genn import Connection, Population, Network
from ml_genn.callbacks import Callback, Checkpoint
from ml_genn.compilers import EPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense, FixedProbability
from ml_genn.initializers import Normal
from ml_genn.neurons import (AdaptiveLeakyIntegrateFire, LeakyIntegrate,
                             LeakyIntegrateFire, SpikeInput)
from ml_genn.serialisers import Numpy
from tonic.datasets import DVSGesture, SHD, SMNIST
from tonic.transforms import Compose, Downsample

from glob import glob
from time import perf_counter
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                log_latency_encode_data, preprocess_tonic_spikes)


class CSVTrainLog(Callback):
    def __init__(self, filename, output_pop):
        # Create CSV writer
        self.file = open(filename, "w")
        self.csv_writer = csv.writer(self.file, delimiter=",")

        # Write header row if we're not resuming from an existing training run
        self.csv_writer.writerow(["Epoch", "Num trials", "Number correct", "Time"])

        self.output_pop = output_pop

    def on_epoch_begin(self, epoch):
        self.start_time = perf_counter()

    def on_epoch_end(self, epoch, metrics):
        m = metrics[self.output_pop]
        self.csv_writer.writerow([epoch, m.total, m.correct, 
                                  perf_counter() - self.start_time])
        self.file.flush()

class Shift:
    def __init__(self, f_shift, sensor_size):
        self.f_shift = f_shift
        self.sensor_size = sensor_size

    def __call__(self, events: np.ndarray) -> np.ndarray:
        # Shift events
        events_copy = copy.deepcopy(events)
        events_copy["x"] = events_copy["x"] + np.random.randint(-self.f_shift, self.f_shift)
        events_copy["y"] = events_copy["y"] + np.random.randint(-self.f_shift, self.f_shift)

        # Delete out of bound events
        events_copy = np.delete(
            events_copy,
            np.where(
                (events_copy["x"] < 0) | (events_copy["x"] >= self.sensor_size[1])
                | (events_copy["y"] < 0) | (events_copy["y"] >= self.sensor_size[0])))
        
        # Return event copy, split into left and right stream
        half_width = self.sensor_size[1] // 2
        left_mask = (events["x"] < half_width)
        left_events = events_copy[left_mask]
        right_events = events_copy[np.logical_not(left_mask)]
        right_events["x"] -= half_width
        return left_events, right_events

class CSVTestLog(Callback):
    def __init__(self, filename, epoch, output_pop):
        # Create CSV writer
        self.file = open(filename, "w")
        self.csv_writer = csv.writer(self.file, delimiter=",")
        self.csv_writer.writerow(["Epoch", "Num trials", "Number correct", "Time"])
        self.epoch = epoch
        self.output_pop = output_pop

    def on_test_begin(self):
        self.start_time = perf_counter()

    def on_test_end(self, metrics):
        m = metrics[self.output_pop]
        self.csv_writer.writerow([self.epoch, m.total, m.correct, 
                                  perf_counter() - self.start_time])
        self.file.flush()

class ConnectivityCheckpoint(Callback):
    def __init__(self, serialiser="numpy"):
        self.serialiser = serialiser

    def set_params(self, compiled_network, **kwargs):
        # Extract compiled network
        self._compiled_network = compiled_network

    def on_epoch_end(self, epoch, metrics):
        # If we should checkpoint this epoch
        self._compiled_network.save_connectivity((epoch,), self.serialiser)



def pad_hidden_layer_argument(arg, num_hidden_layers, context, default=None):
    # If argument wasn't specified but there is a default, repeat default for each hidden layer
    if arg is None and default is not None:
        return [default] * num_hidden_layers
    elif len(arg) == 1:
        return arg * num_hidden_layers
    elif len(arg) != num_hidden_layers:
        raise RuntimeError(f"{context} either needs to be specified as a single "
                           f" value or for each {num_hidden_layers} layers")
    else:
        return arg

def inference(genn_kwargs, args, network, serialiser, latest_spike_time, epoch):
    # Load network state from final checkpoint
    network.load((epoch,), serialiser)

    compiler = InferenceCompiler(evaluate_timesteps=int(np.ceil(latest_spike_time)),
                                 batch_size=1 if args.cpu else args.batch_size, rng_seed=args.seed,
                                 reset_vars_between_batches=False, 
                                 kernel_profiling=args.kernel_profiling, **genn_kwargs)
    compiled_net = compiler.compile(network, name=f"classifier_test_{unique_suffix}")

    with compiled_net:
        # Perform warmup evaluation
        # **TODO** subset of data
        compiled_net.evaluate({input: spikes},
                              {output: labels})

        # Evaluate model on numpy dataset
        start_time = perf_counter()
        callbacks = ["batch_progress_bar",
                     CSVTestLog(f"test_output_{unique_suffix}.csv", epoch, output)]
        metrics, _  = compiled_net.evaluate({input: spikes},
                                            {output: labels},
                                            callbacks=callbacks)
        end_time = perf_counter()
        print(f"Accuracy = {100 * metrics[output].result}%")
        print(f"Time = {end_time - start_time}s")
        
        if args.kernel_profiling:
            print(f"Neuron update time = {compiled_net.genn_model.neuron_update_time}")
            print(f"Presynaptic update time = {compiled_net.genn_model.presynaptic_update_time}")
            print(f"Reset time = {compiled_net.genn_model.get_custom_update_time('Reset')}")

parser = ArgumentParser()
parser.add_argument("--device-id", type=int, default=0, help="CUDA device ID")
parser.add_argument("--train", action="store_true", help="Train model")
parser.add_argument("--cpu", action="store_true", help="Use CPU for inference")
parser.add_argument("--kernel-profiling", action="store_true", help="Output kernel profiling data")
parser.add_argument("--test-all", action="store_true", help="Test all checkpoints up to num epochs")
parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
parser.add_argument("--num-epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--f-target", type=float, default=10.0)
parser.add_argument("--shift", type=int, default=8)
parser.add_argument("--input-sparsity", type=float, default=1.0)
parser.add_argument("--hidden-hidden-sparsity", type=float, default=1.0)
parser.add_argument("--recurrent-sparsity", type=float, default=1.0)

parser.add_argument("--hidden-input-sparsity", type=float, nargs="*")
parser.add_argument("--hidden-recurrent-sparsity", type=float, nargs="*")

args = parser.parse_args()


# Figure out unique suffix for model data
unique_suffix = "_".join(("_".join(str(i) for i in val) if isinstance(val, list) 
                         else str(val))
                         for arg, val in vars(args).items()
                         if arg not in ["train", "cpu", "test_all", "kernel_profiling"])

# Load Tonic datasets
transform = Compose([Downsample(spatial_factor=0.25)])
dataset = DVSGesture(save_to='./data', train=args.train, transform=transform)
sensor_size = (32, 32, 2)

# Get number of input and output neurons from dataset 
# and round up outputs to power-of-two
num_half_input = int(sensor_size[0] * (sensor_size[1] // 2) * sensor_size[2])
num_output = len(dataset.classes)
shift = Shift(args.shift, sensor_size)

# Add events to list
raw_events = []
labels = []
max_spikes = 0
latest_spike_time = 0
for events, label in dataset:
    raw_events.append(events)
    labels.append(label)

    # Calculate max spikes
    # **THINK** shifting could break this
    num_left = sum(events["x"] < 16)
    num_right = len(events) - num_left
    max_spikes = max(max_spikes, num_left, num_right)

    # Calculate max times
    latest_spike_time = max(latest_spike_time, np.amax(events["t"]) / 1000.0)

# Determine max spikes and latest spike time
print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}")

genn_kwargs = {}

serialiser = Numpy("checkpoints_" + unique_suffix)
network = Network()
with network:
    # Add spike input population
    left_input = Population(SpikeInput(max_spikes=args.batch_size * max_spikes),
                            num_half_input)
    right_input = Population(SpikeInput(max_spikes=args.batch_size * max_spikes),
                            num_half_input)

    
    left_ff = Population(AdaptiveLeakyIntegrateFire(v_thresh=0.6, tau_refrac=5.0,
                                                    relative_reset=True, integrate_during_refrac=True),
                         128)
    right_ff = Population(AdaptiveLeakyIntegrateFire(v_thresh=0.6, tau_refrac=5.0,
                                                    relative_reset=True, integrate_during_refrac=True),
                          128)
    
    recurrent = Population(AdaptiveLeakyIntegrateFire(v_thresh=0.6, tau_refrac=5.0,
                                                      relative_reset=True, integrate_during_refrac=True),
                          256)
    
    # Add output population
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="sum_var"),
                        num_output)

    # Add all hidden layer to output layer
    Connection(left_ff, output, Dense(Normal(sd=1.0 / np.sqrt(left_ff.shape[0]))))
    Connection(right_ff, output, Dense(Normal(sd=1.0 / np.sqrt(right_ff.shape[0]))))
    Connection(recurrent, output, Dense(Normal(sd=1.0 / np.sqrt(recurrent.shape[0]))))

    # Connect feedforward layers to input
    in_weight = Normal(sd=1.0 / np.sqrt(num_half_input))
    in_connectivity = (Dense(in_weight) if args.input_sparsity == 1.0 
                       else FixedProbability(args.input_sparsity, in_weight, True))
    Connection(left_input, left_ff, in_connectivity)
    Connection(right_input, right_ff, in_connectivity)

    # Connect hidden layers together
    hid_hid_weight = Normal(sd=1.0 / np.sqrt(128))
    hid_hid_connectivity = (Dense(hid_hid_weight) if args.hidden_hidden_sparsity == 1.0 
                            else FixedProbability(args.hidden_hidden_sparsity, hid_hid_weight))
    Connection(left_ff, recurrent, hid_hid_connectivity)
    Connection(right_ff, recurrent, hid_hid_connectivity)

    # Connect recurrent layer
    rec_weight = Normal(sd=1.0 / np.sqrt(256)) 
    rec_connectivity = (Dense(rec_weight) if args.recurrent_sparsity == 1.0 
                        else FixedProbability(args.args.recurrent_sparsity, rec_weight))
    Connection(recurrent, recurrent, rec_connectivity)

# If we're training model
if args.train:
    assert not args.cpu

    # Create EProp compiler and compile
    compiler = EPropCompiler(example_timesteps=int(np.ceil(latest_spike_time)),
                             losses="sparse_categorical_crossentropy", rng_seed=args.seed,
                             optimiser="adam", batch_size=args.batch_size, f_target=args.f_target,
                             kernel_profiling=args.kernel_profiling, **genn_kwargs)
    compiled_net = compiler.compile(network, name=f"classifier_train_{unique_suffix}")

    with compiled_net:
        # Evaluate model on SHD
        start_time = perf_counter()
        callbacks = ["batch_progress_bar", Checkpoint(serialiser),
                     CSVTrainLog(f"train_output_{unique_suffix}.csv", output),
                     ConnectivityCheckpoint(serialiser)]
        for e in range(args.num_epochs):
            # Loop through raw events
            left_spikes = []
            right_spikes = []
            for r in raw_events:
                # Apply shift augmentation, splitting result into left and right stream
                left_events, right_events = shift(r)
                
                left_spikes.append(preprocess_tonic_spikes(left_events, dataset.ordering,
                                                           (sensor_size[0], sensor_size[1] // 2, sensor_size[2])))
                right_spikes.append(preprocess_tonic_spikes(right_events, dataset.ordering,
                                                           (sensor_size[0], sensor_size[1] // 2, sensor_size[2])))

            # Train epoch
            metrics, _  = compiled_net.train(
                {left_input: left_spikes, right_input: right_spikes}, {output: labels_train},
                start_epoch=e, num_epochs=1, shuffle=True,
                callbacks=callbacks)

        end_time = perf_counter()
        print(f"Accuracy = {100 * metrics[output].result}%")
        print(f"Time = {end_time - start_time}s")
else:
    print(f"Loading inference model from checkpoint {args.num_epochs - 1}")

    # Use CPU backend if desired
    if args.cpu:
        genn_kwargs["backend"]="SingleThreadedCPU"

    # Loop through trained epochs
    if args.test_all:
        for e in range(args.num_epochs):
             inference(genn_kwargs, args, network, serialiser, latest_spike_time, e, e > 0)
    else:
        inference(genn_kwargs, args, network, serialiser, latest_spike_time, args.num_epochs - 1, False)
