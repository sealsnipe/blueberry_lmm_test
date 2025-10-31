# 🚀 Getting Started - Blueberry DE-Train

## One-Command Start

```bash
python start_training.py --device 0
```

That's it! The script handles everything:
- ✅ Setup verification
- ✅ CUDA check
- ✅ Dependency validation
- ✅ Preprocessing (if needed)
- ✅ Training launch

## Full Setup (First Time)

```bash
# 1. Clone repository
git clone https://github.com/sealsnipe/blueberry_lmm_test.git
cd blueberry_lmm_test

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start training!
python start_training.py --device 0 --wandb-mode online
```

## What Happens

1. **Setup Check**: Verifies Python, CUDA, dependencies
2. **Preprocessing**: Loads German datasets (CommonCrawl, OSCAR, etc.)
3. **Training**: Starts 450M PLASA model training
4. **Monitoring**: Live metrics in terminal

## Expected Results (RTX 3090)

- **Throughput**: ~450 tokens/sec
- **VRAM**: ~7GB peak
- **Perplexity**: <8 after 3 epochs
- **Training Time**: ~8-10 hours

## Monitor Training

In a separate terminal:
```bash
python monitoring/live_metrics.py --log_path ./logs/metrics.json
```

## Options

```bash
# Test preprocessing only
python start_training.py --preprocess-only --subset-size 100000

# Use offline logging (no Wandb)
python start_training.py --device 0 --wandb-mode offline

# Skip setup checks (if already verified)
python start_training.py --device 0 --skip-check
```

## Troubleshooting

**OOM Error?** → Reduce batch size in `configs/de_training.yaml`

**No CUDA?** → Script will warn and use CPU (slow but works)

**Wandb Issues?** → Use `--wandb-mode offline`

## Next Steps

After training:
- Check logs in `./logs/`
- View plots in `./logs/plots/`
- Resume with `python resume_training.py`
- Consider LoRA for instruction tuning (v2)

## Documentation

- `README.md` - Complete guide
- `PRODUCTION_LAUNCH.md` - Production details
- `VERIFICATION_REPORT.md` - Test results
- `QUICK_REFERENCE.md` - Command reference

---

**Ready to train! 🚀**

```bash
python start_training.py --device 0
```

