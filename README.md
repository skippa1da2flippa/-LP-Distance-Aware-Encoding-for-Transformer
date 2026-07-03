# LP Distance-Aware Encoding for Transformer

This repository contains the code for an experimental link prediction project based on the master's thesis [Distance-Aware Positional Encoding in Transformers for Improved Link Prediction](https://unitesi.unive.it/retrieve/5ceea8ee-7da3-41e2-b11e-627f3d174599/881423.pdf) by Biagio Barchielli, Ca' Foscari University of Venice, 2024-2025.

The project explores an encoder-decoder architecture for link prediction. For each candidate pair of nodes, the model builds an h-hop subgraph, combines node features with distance-aware positional encodings, processes them with a Transformer encoder, and uses an MLP decoder to estimate the link likelihood.

## Repository Structure

- `custom_blocks/`: reusable neural-network blocks and masked layers.
- `dataset_handler/`: dataset wrappers, dataloader helpers, and split handling.
- `experiments/`: Optuna experiment objectives.
- `models/`: model definitions, decoders, training modules, and scoring utilities.
- `preprocessing/`: graph preprocessing, negative sampling, subgraph extraction, and positional encoding.
- `utils/`: shared helpers for aggregation, saving, GPU checks, and preprocessing pipelines.

## Requirements

- Python 3.13, as specified in `.python-version`.
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management.
- A CUDA-capable PyTorch setup is recommended for model training. CPU execution may work for smaller preprocessing or debugging tasks, but training can be slow.

## Setup with uv

Install `uv` if it is not already available:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Clone or enter the project directory:

```powershell
cd "C:\Users\skippa\Documents\projects\-LP-Distance-Aware-Encoding-for-Transformer"
```

Create or update the virtual environment from the lockfile:

```powershell
uv sync
```

Activate the environment manually when needed:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or run commands directly through `uv` without activating:

```powershell
uv run python main.py
```

## Dependency Notes

Project dependencies are declared in `pyproject.toml` and pinned in `uv.lock`. The main packages include:

- `torch`
- `torch-geometric`
- `lightning`
- `torchmetrics`
- `torcheval`
- `networkx`
- `numpy`
- `optuna`

If PyTorch or CUDA package resolution fails, check the PyTorch index configuration in `pyproject.toml` and make sure it matches the CUDA version available on your machine.

## Typical Workflow

1. Prepare or download the graph dataset with the utilities in `dataset_handler/`.
2. Run preprocessing from `preprocessing/` to create processed splits, subgraphs, positional encodings, and negative samples.
3. Train or evaluate the link prediction models under `models/`.
4. Use the utilities in `models/evaluation/` and `preprocessing/single_hop_multi_process.py` for metrics such as ROC AUC, accuracy, F1, MRR, and Hits@K.

The current `main.py` is a minimal placeholder, so experiments are mainly driven from the project modules rather than a finished command-line interface.

## Thesis Reference

Barchielli, Biagio. *Distance-Aware Positional Encoding in Transformers for Improved Link Prediction*. Master's thesis, Ca' Foscari University of Venice, Department of Environmental Sciences, Informatics and Statistics, 2024-2025.

PDF: <https://unitesi.unive.it/retrieve/5ceea8ee-7da3-41e2-b11e-627f3d174599/881423.pdf>
