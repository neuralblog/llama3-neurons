import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Tokenize and prepare data")
parser.add_argument("--num-processes", type=int, default=16)
parser.add_argument("--data-path", type=str, default="./data")
parser.add_argument("--out-file", type=str, default="./data/tokens.bin")
parser.add_argument(
    "--tokenizer-file", type=str, default="./Meta-Llama-3-8B/original/tokenizer.model"
)

FLAGS = parser.parse_args()


def tokenize_file(file_path: Path):
    import pyarrow.parquet as pq

    from tokenizer import Tokenizer

    enc = Tokenizer(model_path=FLAGS.tokenizer_file)
    df = pq.read_table(file_path, use_threads=None, columns=["text"])
    arr = np.empty((2_000_000_000,), dtype=np.uint32)
    start = 0
    for chunk in df:
        for record in chunk:
            s = record.as_py()
            tokens = enc.encode(s, bos=True, eos=True)
            end = start + len(tokens)
            arr[start:end] = tokens
            start = end
    return (arr, start)


files = sorted(Path(FLAGS.data_path).glob("*_00000.parquet"))

with Pool(processes=FLAGS.num_processes) as pool:
    token_and_size = pool.map(tokenize_file, files)

total_tokens = sum(size for (_, size) in token_and_size)
out_file = Path(FLAGS.out_file)
output = np.memmap(out_file, dtype=np.uint32, mode="w+", shape=(total_tokens,))

offset = 0
for tokens, size in tqdm(token_and_size):
    output[offset : offset + size] = tokens[:size]
    offset += size
output.flush()
