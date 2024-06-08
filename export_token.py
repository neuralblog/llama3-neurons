from argparse import ArgumentParser

import numpy as np
from tqdm.cli import tqdm

from tokenizer import Tokenizer

parser = ArgumentParser()
parser.add_argument("--data-file", default="data/tokens.bin")
parser.add_argument(
    "--tokenizer-file", type=str, default="./Meta-Llama-3-8B/original/tokenizer.model"
)
FLAGS = parser.parse_args()

num_layers = 32
num_neurons_per_layer = 14336
top_start_index = np.load("index.npy")  # layer x neuron x topk
top_seq_act = np.load("act.npy")  # # layer x neuron x topk x seq

data = np.memmap(FLAGS.data_file, dtype=np.uint32, mode="r")
tokenizer = Tokenizer(FLAGS.tokenizer_file)
tokens = np.empty((num_layers, num_neurons_per_layer, 32, 64), dtype=np.uint32)

for layer in tqdm(range(0, num_layers)):
    for neuron in range(num_neurons_per_layer):
        indices = top_start_index[layer, neuron, :]
        batch = []
        for i, s in enumerate(indices):
            e = s + 128
            s = e - 64
            m = s + 32
            a = m - 32
            b = m + 32
            tokens[layer, neuron, i, :] = data[a:b]

np.save("token.npy", tokens)
