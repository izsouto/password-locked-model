# Sandbagging Model

This repository contains code for preprocessing and experiments from the MSc thesis titled 'Picking the Lock: Causal Control,
Mediation and the Anatomy of a
Sandbagging Model'

It implements techniques to investigate sandbagging in password-locked language models, evaluated on the [Hendrycks MATH dataset](https://huggingface.co/datasets/EleutherAI/hendrycks_math).

## Requirements:

Python 3.10, CUDA 12.x, a single GPU with ≥ 40 GB VRAM (experiments used an NVIDIA H200 SXM, 141 GB).

This project also reuses Redwood Research's prompt formatting and MATH grading code, vendored as the `sandbagging` git submodule (see the Pipeline section below).

## Experiment setup

All experiments were conducted on a single NVIDIA H200 SXM GPU (141 GB video
RAM) and were allocated 12 virtual CPUs (vCPUs) and 188 GB of system memory

## Pipeline

Init submodules:
```bash
git submodule update --init --recursive
```

Then run `src/experiments.ipynb` in consecutive order. Each cell prints normalised recovery reported.

## License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0) - see [LICENSE.txt](https://github.com/izsouto/password-locked-model/blob/main/LICENSE.txt) file.
