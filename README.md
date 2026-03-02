# llm-fine-tuning-t4-gpu
Fine-tuning Llama 3.2 3B on a free T4 GPU using LoRA, 4-bit quantization, and Unsloth. Covers GPU memory optimization, mixed-precision training, and inference benchmarking.
# LLM Fine-Tuning on a T4 GPU

Fine-tuned Llama 3.2 3B Instruct using LoRA (rank 16) and 4-bit NF4 quantization 
on a free Google Colab T4 GPU (15GB VRAM).

## Key Results
- Trainable params: 24.3M / 3.2B (0.75%)
- Peak GPU memory: 6.2GB / 14.5GB (42.5% utilization)
- Training time: 327.7s (~5.5 min) for 60 steps
- Throughput: 1.47 samples/sec
- Final loss: 1.1423

## Stack
- Unsloth (2x faster LoRA fine-tuning)
- Hugging Face Transformers + PEFT + TRL
- bitsandbytes (4-bit NF4 quantization)
- 8-bit AdamW optimizer
- Google Colab Tesla T4 GPU (CUDA 12.8)

## What I Learned
- 4-bit quantization reduced model memory ~75% with minimal quality loss
- LoRA adapters trained <1% of total parameters while achieving strong task performance
- Mixed-precision fp16 leveraged T4 Tensor Cores for faster matrix ops
- 8-bit Adam optimizer cut optimizer state memory by 50%
- At 42.5% GPU utilization, there was headroom to increase batch size or sequence length for higher throughput
- Gradient accumulation (4 steps) simulated batch size of 8 on limited VRAM

## Run It
Open the notebook in Google Colab, set runtime to T4 GPU, and run all cells.
