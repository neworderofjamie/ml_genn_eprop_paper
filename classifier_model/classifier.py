import csv
import numpy as np

from argparse import ArgumentParser
from pygenn.genn_wrapper.CUDABackend import DeviceSelect_MANUAL
from ml_genn import Connection, Population, Network
from ml_genn.callbacks import Callback, Checkpoint
from ml_genn.compilers import InferenceCompiler
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import (AdaptiveLeakyIntegrateFire, LeakyIntegrate,
                             LeakyIntegrateFire, SpikeInput)
from ml_genn.serialisers import Numpy
from ml_genn_eprop import EPropCompiler

from time import perf_counter
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                log_latency_encode_data, preprocess_tonic_spikes)


class CSVTrainLog(Callback):
    def __init__(self, filename, output_pop):
        # Create CSV writer
        self.file = open(filename, "w")
        self.csv_writer = csv.writer(self.file, delimiter=",")
        self.csv_writer.writerow(["Epoch", "Num trials", "Number correct", "Time"])
        self.output_pop = output_pop
    
    def on_epoch_begin(self, epoch):
        self.start_time = perf_counter()

    def on_epoch_end(self, epoch, metrics):
        m = metrics[self.output_pop]
        self.csv_writer.writerow([epoch, m.total, m.correct, 
                                  perf_counter() - self.start_time])
        self.file.flush()

class CSVTestLog(Callback):
    def __init__(self, filename, output_pop):
        # Create CSV writer
        self.file = open(filename, "w")
        self.csv_writer = csv.writer(self.file, delimiter=",")
        self.csv_writer.writerow(["Num trials", "Number correct", "Time"])
        self.output_pop = output_pop
    
    def on_test_begin(self):
        self.start_time = perf_counter()

    def on_test_end(self, metrics):
        m = metrics[self.output_pop]
        self.csv_writer.writerow([m.total, m.correct, 
                                  perf_counter() - self.start_time])
        self.file.flush()
                                  
def pad_hidden_layer_argument(arg, num_hidden_layers, context):
    if len(arg) == 1:
        return arg * num_hidden_layers
    elif len(arg) != num_hidden_layers:
        raise RuntimeError(f"{context} either needs to be specified as a single "
                           f" value or for each {num_hidden_layers} layers")
    else:
        return arg


parser = ArgumentParser()
parser.add_argument("--device-id", type=int, default=0, help="CUDA device ID")
parser.add_argument("--train", action="store_true", help="Train model")
parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
parser.add_argument("--num-epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--dataset", choices=["smnist", "shd", "dvs_gesture", "mnist"], required=True)
parser.add_argument("--seed", type=int, default=1234)

parser.add_argument("--hidden-size", type=int, nargs="*")
parser.add_argument("--hidden-recurrent", choices=["True", "False"], nargs="*")
parser.add_argument("--hidden-model", choices=["lif", "alif"], nargs="*")

args = parser.parse_args()

num_hidden_layers = max(len(args.hidden_size), 
                        len(args.hidden_recurrent),
                        len(args.hidden_model))
print(f"{num_hidden_layers} hidden layers")

# Pad hidden layer arguments
args.hidden_size = pad_hidden_layer_argument(args.hidden_size, 
                                             num_hidden_layers,
                                             "Hidden layer size")
args.hidden_recurrent = pad_hidden_layer_argument(args.hidden_recurrent, 
                                                  num_hidden_layers,
                                                  "Hidden layer recurrentness")
args.hidden_model = pad_hidden_layer_argument(args.hidden_model, 
                                              num_hidden_layers,
                                              "Hidden layer neuron model")

# Figure out unique suffix for model data
unique_suffix = "_".join(("_".join(str(i) for i in val) if isinstance(val, list) 
                         else str(val))
                         for arg, val in vars(args).items() if arg != "train")

# If dataset is MNIST
spikes = []
labels = []
num_input = None
num_output = None
if args.dataset == "mnist":
    import mnist
    
    # Latency encode MNIST digits
    num_input = 28 * 28
    num_output = 10
    labels = mnist.train_labels() if args.train else mnist.test_labels()
    spikes = log_latency_encode_data(
        mnist.train_images() if args.train else mnist.test_images(),
        20.0, 51)
# Otherwise
else:
    from tonic.datasets import DVSGesture, SHD, SMNIST
    from tonic.transforms import Compose, Downsample

    # Load Tonic datasets
    if args.dataset == "shd":
        dataset = SHD(save_to='./data', train=args.train)
        sensor_size = dataset.sensor_size
    elif args.dataset == "smnist":
        dataset = tonic.datasets.SMNIST(save_to='./data', train=args.train, 
                                        duplicate=False, num_neurons=79)
        sensor_size = dataset.sensor_size
    elif args.dataset == "dvs_gesture":
        transform = Compose([Downsample(spatial_factor=0.25)])
        dataset = DVSGesture(save_to='./data', train=args.train, transform=transform)
        sensor_size = (32, 32, 2)
    
    # Get number of input and output neurons from dataset 
    # and round up outputs to power-of-two
    num_input = int(np.prod(sensor_size))
    num_output = len(dataset.classes)

    # Preprocess spike
    for events, label in dataset:
        spikes.append(preprocess_tonic_spikes(events, dataset.ordering,
                                              sensor_size))
        labels.append(label)

# Determine max spikes and latest spike time
max_spikes = calc_max_spikes(spikes)
latest_spike_time = calc_latest_spike_time(spikes)
print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}")

genn_kwargs = {"selectGPUByDeviceID": True,
               "deviceSelectMethod": DeviceSelect_MANUAL,
               "manualDeviceID": args.device_id}

serialiser = Numpy("checkpoints_" + unique_suffix)
network = Network()
with network:
    # Add spike input population
    input = Population(SpikeInput(max_spikes=args.batch_size * max_spikes),
                       num_input)
    
    # Loop through hidden layers
    hidden = []
    for i, (s, r, m) in enumerate(zip(args.hidden_size, 
                                      args.hidden_recurrent,
                                      args.hidden_model)):
        # Add population
        if m == "alif":
            hidden.append(Population(AdaptiveLeakyIntegrateFire(v_thresh=0.6,
                                                                tau_refrac=5.0,
                                                                relative_reset=True,
                                                                integrate_during_refrac=True),
                                     s))
        else:
            hidden.append(Population(LeakyIntegrateFire(v_thresh=0.61, tau_mem=20.0,
                                                        tau_refrac=5.0,
                                                        relative_reset=True,
                                                        integrate_during_refrac=True),
                                     s))
        
        # If recurrent, add recurrent connections
        if r == "True":
            Connection(hidden[-1], hidden[-1], 
                       Dense(Normal(sd=1.0 / np.sqrt(s))))
       
        # If this is first hidden layer, add input connections
        if i == 0:
            Connection(input, hidden[-1], Dense(Normal(sd=1.0 / np.sqrt(num_input))))
        # Otherwise, add connection to previous hidden layer
        else:
            Connection(hidden[-2], hidden[-1], Dense(Normal(sd=1.0 / np.sqrt(hidden[-2].shape[0]))))
    
    # Add output population
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="sum_var", softmax=True),
                        num_output)

    # Add connection to last hidden layer
    Connection(hidden[-1], output, Dense(Normal(sd=1.0 / np.sqrt(hidden[-1].shape[0]))))

# If we're training model
if args.train:
    # Create EProp compiler and compile
    compiler = EPropCompiler(example_timesteps=int(np.ceil(latest_spike_time)),
                             losses="sparse_categorical_crossentropy", rng_seed=args.seed,
                             optimiser="adam", batch_size=args.batch_size, **genn_kwargs)
    compiled_net = compiler.compile(network, name=f"classifier_train_{unique_suffix}")

    with compiled_net:
        # Evaluate model on SHD
        start_time = perf_counter()
        callbacks = ["batch_progress_bar", Checkpoint(serialiser),
                     CSVTrainLog(f"train_output_{unique_suffix}.csv", output)]
        metrics, _  = compiled_net.train({input: spikes},
                                         {output: labels},
                                         num_epochs=args.num_epochs,
                                         callbacks=callbacks,
                                         shuffle=True)
        end_time = perf_counter()
        print(f"Accuracy = {100 * metrics[output].result}%")
        print(f"Time = {end_time - start_time}s")
else:
    # Load network state from final checkpoint
    network.load((args.num_epochs - 1,), serialiser)

    compiler = InferenceCompiler(evaluate_timesteps=int(np.ceil(latest_spike_time)),
                                 batch_size=args.batch_size, rng_seed=args.seed, 
                                 reset_vars_between_batches=False, **genn_kwargs)
    compiled_net = compiler.compile(network, name=f"classifier_test_{unique_suffix}")

    with compiled_net:
        # Perform warmup evaluation
        # **TODO** subset of data
        compiled_net.evaluate({input: spikes},
                              {output: labels})
                                            
        # Evaluate model on numpy dataset
        start_time = perf_counter()
        callbacks = ["batch_progress_bar",
                     CSVTestLog(f"test_output_{unique_suffix}.csv", output)]
        metrics, _  = compiled_net.evaluate({input: spikes},
                                            {output: labels},
                                            callbacks=callbacks)
        end_time = perf_counter()
        print(f"Accuracy = {100 * metrics[output].result}%")
        print(f"Time = {end_time - start_time}s")
