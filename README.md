# llm-fine-tuning-t4-gpu
Fine-tuning Llama 3.2 3B on a free T4 GPU using LoRA, 4-bit quantization, and Unsloth. Covers GPU memory optimization, mixed-precision training, and inference benchmarking.
# LLM Fine-Tuning on a T4 GPU

Fine-tuned Llama 3.2 3B Instruct using LoRA (rank 16) and 4-bit NF4 quantization 
on a free Google Colab T4 GPU (15GB VRAM).

## Key Results
- Trainable params: 24.3M / 3.2B (0.75%)
- Peak GPU memory: ~8GB / 14.5GB
- Training time: ~X minutes for 60 steps
- Final loss: X.XXXX

## Stack
- Unsloth (2x faster LoRA fine-tuning)
- Hugging Face Transformers + PEFT + TRL
- bitsandbytes (4-bit quantization)
- Google Colab T4 GPU

## What I Learned
- 4-bit quantization reduces model memory ~75% with minimal quality loss
- LoRA adapters train <1% of total parameters while achieving strong task performance
- Mixed-precision (fp16) leverages T4 Tensor Cores for ~2x speedup
- 8-bit Adam optimizer cuts optimizer state memory by 50%
- Gradient accumulation lets you simulate larger batch sizes on limited VRAM

## Run It
Open the notebook in Google Colab, set runtime to T4 GPU, and run all cells.
