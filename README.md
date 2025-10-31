# German PLASA LLM Training

A complete training pipeline for a German language model (~450M parameters) based on the Blueberry LLM framework with PLASA (DeepSeek Sparse) Attention. Designed to run efficiently on a single NVIDIA RTX 3090 (24GB VRAM) with FP16/FP4 quantization support.

## Features

- **PLASA Attention**: DeepSeek Sparse Attention mechanism for efficient long-sequence processing
- **Multi-Dataset Support**: Streaming support for multiple German datasets (CommonCrawl, OSCAR, FineWeb, Wikipedia, Alpaca-German)
- **Quantization**: FP16 and FP4 quantization support via bitsandbytes
- **Gradient Accumulation**: Efficient training with gradient accumulation
- **Mixed Precision**: Automatic mixed precision training
- **Checkpointing**: Automatic checkpoint saving and resuming
- **Live Monitoring**: Asynchronous metrics monitoring with real-time display
- **Comprehensive Logging**: Wandb integration + local JSON/CSV logging
- **Visualization**: Automatic plot generation for training curves

## Project Structure

```
.
├── configs/
│   ├── de_training.yaml       # Main training configuration
│   └── __init__.py
├── data/
│   ├── preprocess.py          # Dataset preprocessing with German filtering
│   └── __init__.py
├── models/
│   ├── plasa_model.py         # PLASA attention model implementation
│   └── __init__.py
├── training/
│   ├── de_train.py            # Main training script
│   └── __init__.py
├── monitoring/
│   ├── live_metrics.py        # Live metrics monitoring script
│   └── __init__.py
├── utils/
│   ├── logging.py             # Logging utilities
│   ├── metrics.py             # Metrics tracking and visualization
│   └── __init__.py
├── tests/
│   ├── test_preprocess.py     # Unit tests
│   ├── test_integration.py    # Integration tests
│   └── conftest.py
├── logs/                      # Training logs and metrics
├── checkpoints/               # Model checkpoints
├── requirements.txt           # Python dependencies
├── run_full.py               # Full pipeline wrapper
└── README.md

```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 12+ (for GPU training)
- NVIDIA RTX 3090 (or compatible GPU with 24GB+ VRAM)

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd blueberrylmm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Setup Wandb for logging:
```bash
wandb login
```

## Quick Start

### Basic Training

Run the full training pipeline with monitoring:

```bash
python run_full.py --config configs/de_training.yaml --device 0
```

### Training Only (No Monitoring)

```bash
python training/de_train.py --config configs/de_training.yaml --device 0 --wandb_mode online
```

### Monitoring Only

In a separate terminal:

```bash
python monitoring/live_metrics.py --log_path ./logs/metrics.json --interval 30
```

### Resume Training

**Option 1: Using the resume helper script (recommended):**
```bash
python resume_training.py --config configs/de_training.yaml --device 0
# Automatically finds latest checkpoint
```

**Option 2: Specify checkpoint manually:**
```bash
python run_full.py --config configs/de_training.yaml --device 0 --resume_from ./checkpoints/latest.pt
```

**Option 3: Direct training script:**
```bash
python training/de_train.py --config configs/de_training.yaml --device 0 --resume_from ./checkpoints/checkpoint_step_1000.pt
```

## Configuration

Edit `configs/de_training.yaml` to customize training:

### Model Architecture

```yaml
model:
  vocab_size: 50257          # GPT-2 tokenizer vocab size
  d_model: 1024             # Model dimension
  n_heads: 16               # Attention heads
  n_layers: 24              # Transformer layers
  d_ff: 4096                # Feed-forward dimension
  max_seq_len: 1024         # Maximum sequence length
  
  attention:
    type: "plasa"           # PLASA attention
    indexer_heads: 4        # Indexer heads
    indexer_dim: 64         # Indexer dimension
    sparse_top_k: 512       # Top-k tokens for sparse attention
```

### Training Settings

```yaml
training:
  batch_size: 32
  gradient_accumulation_steps: 4
  learning_rate: 5e-4
  precision: "fp16"        # Options: fp32, fp16, fp4
  fp4_enabled: false        # Enable FP4 quantization
  max_epochs: 3
  eval_every: 500          # Evaluation frequency
  save_every: 100          # Checkpoint frequency
```

### Dataset Configuration

```yaml
dataset:
  datasets:
    - name: "commoncrawl"
      path: "commoncrawl/de"
      weight: 0.60
      streaming: true
    # ... more datasets
  
  preprocessing:
    min_length_tokens: 256
    max_length_tokens: 1024
    target_length_tokens: 512
    language_filter: "de"
```

## Datasets

The pipeline supports multiple German datasets with streaming:

1. **CommonCrawl (de)**: 60% weight
2. **OSCAR (de)**: 20% weight
3. **FineWeb-Edu (de-filtered)**: 10% weight
4. **German Wikipedia**: 5% weight
5. **Alpaca-German**: 5% weight

All datasets are loaded in streaming mode to avoid full downloads. The preprocessor automatically:
- Filters for German text using `langdetect`
- Deduplicates texts
- Tokenizes using GPT-2 tokenizer
- Chunks into sequences of target length

## Monitoring

### Live Metrics Display

The monitoring script displays real-time metrics:

```
╔════════════════════════════════════════════════════════════════╗
║  Training Metrics Monitor - 2024-01-01 12:00:00              ║
╠════════════════════════════════════════════════════════════════╣
║  Step: 1000       Epoch: 1                                     ║
║  Loss: 2.1000 ↓   Perplexity: 8.17 ↓                          ║
║  Learning Rate: 4.50e-04                                       ║
║  VRAM Usage: 18.50 GB                                         ║
║  Tokens/sec: 500 ↑                                             ║
╚════════════════════════════════════════════════════════════════╝
```

### Metrics File

Metrics are saved to `./logs/metrics.json` in JSON format:

```json
[
  {
    "step": 100,
    "epoch": 0,
    "loss": 2.5,
    "perplexity": 12.18,
    "learning_rate": 5e-4,
    "vram_usage": 18.5,
    "tokens_per_second": 500
  }
]
```

### Visualization

After training, generate plots:

```python
from utils.metrics import plot_comprehensive_metrics

plot_comprehensive_metrics(
    metrics_file="./logs/metrics.json",
    output_path="./logs/plots/training_curves.png"
)
```

## Hardware Requirements

### Minimum (Testing)
- GPU: NVIDIA RTX 3090 (24GB VRAM)
- RAM: 32GB
- Storage: 100GB+ for datasets

### Recommended (Production)
- GPU: NVIDIA RTX 3090 or better
- RAM: 64GB+
- Storage: 500GB+ SSD

## Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size in config:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

2. Increase gradient accumulation:
```yaml
training:
  gradient_accumulation_steps: 8  # Increase from 4
```

3. Enable FP4 quantization:
```yaml
training:
  precision: "fp4"
  fp4_enabled: true
```

### FP4 Setup Issues

If FP4 quantization fails:

1. Ensure bitsandbytes is installed:
```bash
pip install bitsandbytes
```

2. Check CUDA compatibility:
```bash
python -c "import bitsandbytes; print(bitsandbytes.__version__)"
```

3. Fall back to FP16:
```yaml
training:
  precision: "fp16"
  fp4_enabled: false
```

### Dataset Loading Issues

If datasets fail to load:

1. Check internet connection (for streaming)
2. Verify dataset names/paths in config
3. Try loading datasets individually:
```python
from datasets import load_dataset
ds = load_dataset("commoncrawl", "de", streaming=True)
```

### Wandb Issues

If Wandb logging fails:

1. Login to Wandb:
```bash
wandb login
```

2. Use offline mode:
```bash
python run_full.py --config configs/de_training.yaml --wandb_mode offline
```

3. Disable Wandb:
```yaml
logging:
  wandb:
    enabled: false
```

## Testing

Run unit tests:

```bash
pytest tests/test_preprocess.py -v
```

Run integration tests:

```bash
pytest tests/test_integration.py -v
```

Run all tests:

```bash
pytest tests/ -v
```

## Performance Tuning

### VRAM Optimization

- Use FP16 instead of FP32 (saves ~50% VRAM)
- Use FP4 for even more savings (saves ~75% VRAM)
- Reduce batch size or sequence length
- Enable gradient checkpointing (if implemented)

### Speed Optimization

- Increase batch size (if VRAM allows)
- Use gradient accumulation to simulate larger batches
- Optimize data loading (num_workers, pin_memory)
- Use mixed precision training

## Model Architecture Details

### PLASA Attention

The model uses DeepSeek Sparse Attention (PLASA) with:
- Lightning Indexer for efficient token selection
- Top-k sparse attention (configurable k)
- Causal masking for autoregressive generation

### Model Size

- **Total Parameters**: ~450M
- **Layers**: 24
- **Hidden Dimension**: 1024
- **Attention Heads**: 16
- **Feed-Forward Dimension**: 4096

## Citation

If you use this codebase, please cite:

```bibtex
@misc{blueberry-llm-de,
  title={German PLASA LLM Training Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## License

[Specify your license]

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Acknowledgments

- Blueberry LLM Framework: https://github.com/Open-Superintelligence-Lab/blueberry-llm
- DeepSeek Sparse Attention
- HuggingFace Transformers and Datasets

## GitHub Setup

For detailed GitHub launch instructions, see:
- `GITHUB_LAUNCH_CHECKLIST.md`: Complete launch checklist
- `GITHUB_READY.md`: Final validation summary  
- `setup_github.py`: Helper script for GitHub setup

Quick setup:
```bash
git init
git add .
git commit -m "Initial commit: German PLASA LLM training pipeline"
git remote add origin https://github.com/YOUR_USERNAME/blueberry-de-train.git
git push -u origin main
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the configuration documentation

## Additional Resources

- `QUICK_REFERENCE.md`: Quick command reference and troubleshooting
- `VALIDATION_STATUS.md`: Test results and validation summary
- `PACKAGE_SUMMARY.md`: Complete package overview and quick reference
- `resume_training.py`: Helper script for resuming training
- `quick_start.py`: Quick start script with dependency checks

---

**Happy Training! 🚀**

