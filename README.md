# Llama3-8B MLP Neurons

This codebase is used to collect text snippets that strongly activate MLP neurons in the Llama3-8B model.

**Note**: If you don't want to download the neuron data (which is likely the case), add the `--single-branch` flag to your `git clone` command.

## Data Preparation

We use the FineWeb-Edu dataset (`sample-10BT` subset). Use the following commands to download the dataset to the `./data` local directory:

```bash
mkdir -p data
cat urls.txt | xargs -n 1 -P 14 wget -P data
```

To prepare the tokens, we first need to download the Llama3 tokenizer:

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B --include "original/tokenizer.model" --local-dir Meta-Llama-3-8B
```

Then, run the following script to prepare the token data. Note that it will consume a lot of RAM. Reduce the number of parallel processes to fit within your system's RAM capacity.

```bash
python prepare_data.py --num-processes=16
```

The script creates a single binary file at `./data/tokens.bin`.

## Collecting Neuron Activations

Now that we have the data ready, we can start collecting the top activations for each MLP neuron. First, we need to download the model:

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B --include "original/*" --local-dir Meta-Llama-3-8B
```

Then, run:

```bash
python find_top_activations.py
```

It takes approximately 12 hours on an A100 GPU to process 4M snippets. The results are two files: `act.npy` and `index.npy`, which store the top activations for each neuron and the start indices of the corresponding text snippets.

## Exporting to Neuron Viewer Format

To export data to the neuron viewer format:

```bash
python export_neuron.py
```

To start an HTTP server:
```bash
python -m http.server --directory ./docs
```

Visit http://127.0.0.1:8000/neuron_viewer.html to start exploring MLP neurons!