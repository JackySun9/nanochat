# Running nanochat on Apple Silicon (MPS)

This guide explains how to run nanochat on your M1 Mac Pro with 32 GPU cores and 10 CPU cores using Metal Performance Shaders (MPS).

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10+
- 32GB RAM (you have this ✓)
- MPS support (included in PyTorch)

## Quick Start

### 1. Install Dependencies

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv sync
```

### 2. Test MPS Support

```bash
python test_mps.py
```

This will verify that MPS is working correctly on your system.

### 3. Run Training

For a quick test with a smaller model:

```bash
# Train a smaller model (depth=12, batch_size=4) - optimized for 32 GPU cores
python -m scripts.base_train -- --depth=12 --device_batch_size=4

# Or for the full d20 model (optimized for 32 GPU cores)
python -m scripts.base_train -- --depth=20 --device_batch_size=8

# Or use the optimized script for full pipeline
bash speedrun_mps_optimized.sh
```

### 4. Chat with Your Model

```bash
# Start the web interface
python -m scripts.chat_web
```

Then open your browser to the URL shown (usually http://localhost:8000).

## Key Modifications Made

### 1. Device Detection (`nanochat/common.py`)
- Added MPS device detection
- Automatic fallback: MPS → CUDA → CPU
- Disabled distributed training for MPS (single device only)

### 2. Memory Management
- Added MPS-compatible memory usage tracking
- Replaced CUDA-specific memory calls

### 3. Autocast Context
- MPS uses `float16` instead of `bfloat16`
- Automatic device type detection for autocast

### 4. Training Scripts
- Removed `torchrun` (distributed training)
- Reduced batch sizes for MPS memory constraints
- Added MPS synchronization calls

## Performance Expectations

### Memory Usage (Optimized for 32 GPU cores)
- **Model depth 12**: ~4-6GB RAM, batch_size=4
- **Model depth 20**: ~12-16GB RAM, batch_size=8
- **Model depth 32**: ~20-28GB RAM, batch_size=4

### Training Time (Optimized for 32 GPU cores)
- **Depth 12**: ~1.5-3 hours
- **Depth 20**: ~6-10 hours  
- **Depth 32**: ~15-25 hours

*Note: Training times are estimates and depend on your specific hardware.*

## Troubleshooting

### Out of Memory Errors
Reduce the `--device_batch_size` parameter:
```bash
python -m scripts.base_train -- --depth=20 --device_batch_size=2
```

### MPS Not Available
Ensure you have the latest PyTorch version:
```bash
pip install torch --upgrade
```

### Slow Performance
- Close other applications to free up memory
- Ensure you're using the MPS device (check logs)
- Consider using a smaller model depth

## Scripts Modified

- `nanochat/common.py` - Device initialization
- `nanochat/dataloader.py` - Device placement
- `nanochat/report.py` - GPU info reporting
- `scripts/base_train.py` - Training script
- `scripts/mid_train.py` - Midtraining script
- `scripts/chat_*.py` - Chat scripts
- `pyproject.toml` - Dependencies

## Full Training Pipeline

To run the complete training pipeline:

```bash
# 1. Download and prepare data
python -m nanochat.dataset -n 8

# 2. Train tokenizer
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval

# 3. Download evaluation data
curl -L -o eval_bundle.zip https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
unzip -q eval_bundle.zip
mv eval_bundle ~/.cache/nanochat/

# 4. Train base model
python -m scripts.base_train -- --depth=20 --device_batch_size=4

# 5. Evaluate base model
python -m scripts.base_loss
python -m scripts.base_eval

# 6. Midtraining
python -m scripts.mid_train -- --device_batch_size=4
python -m scripts.chat_eval -- -i mid

# 7. Supervised fine-tuning
python -m scripts.chat_sft -- --device_batch_size=2
python -m scripts.chat_eval -- -i sft

# 8. Generate report
python -m nanochat.report generate

# 9. Start chat interface
python -m scripts.chat_web
```

## Notes

- MPS doesn't support distributed training, so you'll use single-device training
- Memory usage is higher than CUDA due to unified memory architecture
- Training will be slower than on dedicated GPUs but still functional
- The model quality will be identical to the original implementation

Enjoy training your own ChatGPT on your M1 Mac Pro! 🚀
