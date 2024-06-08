import json
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
top_start_index = np.load("index.npy")
top_start_index = np.reshape(top_start_index, (-1, num_neurons_per_layer, num_layers))
top_seq_act = np.load("act.npy")
assert top_seq_act.shape[0] == 32  # top 32
assert top_seq_act.shape[1] == 64  # seq len 64
top_seq_act = np.reshape(top_seq_act, (32, 64, num_neurons_per_layer, num_layers))

data = np.memmap(FLAGS.data_file, dtype=np.uint32, mode="r")
tokenizer = Tokenizer(FLAGS.tokenizer_file)

for layer in range(0, num_layers):
    for chunk_start in range(0, top_start_index.shape[1], 1000):
        chunk_end = min(chunk_start + 1000, top_start_index.shape[1])
        layer_data = {}
        for neuron in tqdm(range(chunk_start, chunk_end)):
            indices = top_start_index[:, neuron, layer]
            batch = []
            for s in indices:
                e = s + 128
                s = e - 64
                m = s + 32
                a = m - 16
                b = m + 16
                batch.append(data[a:b])

            tokens = np.stack(batch, axis=0)
            act = top_seq_act[:, :, neuron, layer]
            act = act - act.min()
            act = act / act.max() * 10
            act = act.astype(np.int32)
            m = 32
            a = m - 16
            b = m + 16
            act = act[:, a:b]
            neuron_data = []
            for i in range(tokens.shape[0]):
                text = [
                    w.decode("utf-8", errors="ignore")
                    for w in tokenizer.model.decode_tokens_bytes(tokens[i])
                ]
                activations = act[i].tolist()
                neuron_data.append({"tokens": text, "activations": activations})
            layer_data[neuron] = neuron_data

        json.dump(
            layer_data,
            open(
                f"docs/neuron_viewer_data/layer_{layer}_chunk_{chunk_start}.json", "w"
            ),
        )
