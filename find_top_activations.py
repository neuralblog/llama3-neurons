import json
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from tqdm.cli import tqdm

from model import ModelArgs, Transformer

parser = ArgumentParser()
parser.add_argument("--batch-size", default=32, type=int, required=False)
parser.add_argument("--num-examples", default=4_000_000, type=int, required=False)
parser.add_argument("--data-file", default="data/tokens.bin", type=str, required=False)
parser.add_argument(
    "--ckpt-path", default="Meta-Llama-3-8B/original", type=Path, required=False
)
parser.add_argument("--num-top-examples", default=32, type=int, required=False)

FLAGS = parser.parse_args()


class TopActivation:
    def __init__(self, largest):
        super().__init__()
        self.largest = largest
        self.top_act = None
        self.top_seq_act = None
        self.top_start_index = None

    def __call__(self, seq_act, start_index):
        act = seq_act[:, 32]

        if self.top_act is not None:
            act = torch.concat((act, self.top_act), dim=0)
            seq_act = torch.concat((seq_act, self.top_seq_act), dim=0)
            start_index = torch.concat((start_index, self.top_start_index), dim=0)

        top_act, top_idx = torch.topk(act, k=32, dim=0, largest=self.largest)

        top_start_index = torch.gather(start_index, 0, top_idx)
        top_idx = torch.broadcast_to(
            top_idx[:, None, :], (top_idx.shape[0], seq_act.shape[1], top_idx.shape[1])
        )
        top_seq_act = torch.gather(seq_act, 0, top_idx)
        self.top_act = top_act
        self.top_start_index = top_start_index
        self.top_seq_act = top_seq_act


def get_batch_data(data, bs, seq_len):
    rng = random.Random(42)
    examples = []
    end = data.shape[0] - seq_len
    indices = []
    while True:
        i = rng.randint(0, end)
        j = i + seq_len
        indices.append(i)
        examples.append(data[i:j])

        if len(examples) == bs:
            yield np.array(examples), np.array(indices)
            indices = []
            examples = []


torch.set_grad_enabled(False)
torch.set_default_device("cuda")
torch.set_default_dtype(torch.bfloat16)

batch_size = FLAGS.batch_size
total_examples = FLAGS.num_examples
data = np.memmap(FLAGS.data_file, dtype=np.uint32, mode="r")


data_iter = get_batch_data(data, batch_size, 128)
num_iters = total_examples // batch_size
total_examples = num_iters * batch_size


with open(FLAGS.ckpt_path / "params.json", "r") as f:
    params = json.loads(f.read())

model_args: ModelArgs = ModelArgs(
    max_seq_len=128,
    max_batch_size=batch_size,
    **params,
)
model = Transformer(model_args)

ckpt = FLAGS.ckpt_path / "consolidated.00.pth"
checkpoint = torch.load(ckpt, map_location="cpu", mmap=True)

model.load_state_dict(checkpoint, strict=True)
model = torch.compile(model, fullgraph=True)
del checkpoint

top_act = TopActivation(largest=True)

for it, (token_, start_index_) in tqdm(
    zip(range(num_iters), data_iter), total=num_iters
):
    token = torch.tensor(token_, dtype=torch.long)
    start_index = torch.tensor(start_index_, dtype=torch.long)
    seq_act = model(token)
    seq_act = seq_act.reshape(batch_size, seq_act.shape[1], -1)
    start_index = torch.repeat_interleave(start_index[:, None], 458752, 1)
    top_act(seq_act, start_index)

index = top_act.top_start_index
act = top_act.top_seq_act

index = index.reshape(index.shape[0], 14336, 32)  # topk x neuron x layer
act = act.reshape(act.shape[0], act.shape[1], 14336, 32)  # topk x seq x neuron x layer

index = index.permute(2, 1, 0)  # layer x neuron x topk
act = act.permute(3, 2, 0, 1)  # layer x neuron x topk x seq

np.save("index.npy", index.cpu().numpy())
np.save("act.npy", act.float().cpu().numpy())
