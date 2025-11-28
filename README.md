# SkyRL Demo

Colab-friendly notebook that fine-tunes a small Qwen2.5 model on the GSM8K math dataset with SkyRL using GRPO and vLLM, all inside a lightweight `uv` environment.

## Contents
- `SkyRL.ipynb`: End-to-end workflow—env setup, data prep, Ray startup, and training run.
- `LICENSE`: Repository license.

## Prerequisites
- Google Colab (GPU runtime recommended; vLLM needs GPU).
- GitHub access to clone `NovaSky-AI/SkyRL`.

## How to run
1) Open `SkyRL.ipynb` in Colab (use the badge in the first cell).  
2) Run cells top-to-bottom:
   - Install `uv`, clone the SkyRL repo, and set the Ray runtime hook so Ray uses `uv` for envs.
   - Build GSM8K train/validation Parquet files via `examples/gsm8k/gsm8k_dataset.py` into `/content/data/gsm8k`.
   - Start a local Ray head node.
   - Generate and execute `run_gsm8k_colab.sh`, which launches training with:
     - Base model: `Qwen/Qwen2.5-0.5B-Instruct`
     - Algorithm: GRPO; strategy: FSDP2; backend: vLLM; GPU count set by `NUM_GPUS` (default 1)
     - Data: `/content/data/gsm8k/train.parquet` and `validation.parquet`
     - Key knobs: batch/microbatch sizes, KL loss on, max prompt/gen lengths 512/256, sampling params, and checkpoint path `/content/ckpts/gsm8k_0_5B`
3) Watch console logs for training progress and checkpoints; tweak script flags as needed (e.g., `NUM_GPUS`, sampling settings, learning rate).

## Why this exists
- Provides a reproducible, minimal recipe to exercise SkyRL on GSM8K before scaling up.
- Uses `uv` for clean, isolated dependency management in Colab.
- Shows how to pair Ray + vLLM for RL-style fine-tuning with a small, fast model.
