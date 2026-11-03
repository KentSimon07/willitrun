# willitrun

Check which AI models your PC can actually run, and how fast. Auto-detects your GPU, VRAM & RAM. 100% offline.

```
GPU: Intel(R) Arc(TM) 140T GPU (15.7 GB, integrated)   RAM: 31.4 GB

 Model              Quant   Needs     Status        Speed       Run with
 GPT-OSS 20B        Q4      14.5 GB   ✅ Fits       ~28 tok/s   ollama run gpt-oss:20b
 Qwen3 14B          Q4      10.2 GB   ✅ Fits       ~7 tok/s    ollama run qwen3:14b
 GLM 4.7 Flash      Q4      20.7 GB   🐢 CPU only   ~20 tok/s   ollama run glm-4.7-flash
 Llama 3.3 70B      Q4      48.7 GB   ❌ Too big    -           ollama run llama3.3:70b

Best for you: Qwen3 30B A3B Q4 (~18 tok/s) → ollama run qwen3:30b
```

## Install & run

Requires Python 3.9+.

```
git clone https://github.com/kenttsimon/willitrun
cd willitrun
python -m pip install -r requirements.txt
python willitrun.py
```

## How it works

1. **Detects your hardware.** NVIDIA via `nvidia-smi`, Intel/AMD on Windows, and Apple Silicon.
2. **Checks if each model fits.** Size ≈ parameters × bits per weight, plus about 15% for context and runtime. It tries Q8 first, then Q4.
3. **Estimates speed.** Text generation is limited by memory bandwidth, so tokens/sec ≈ bandwidth ÷ model size.
4. **Handles MoE models.** Mixture-of-experts models only read their active parameters per token, so a 30B model with 3B active runs about as fast as a 3B model.

Speeds are estimates. Real results depend on drivers, context length and background apps.

## Safe to run

- **Offline.** No internet calls, your specs never leave your machine.
- **Read-only.** It only reads hardware info and never changes files or settings.
- **Minimal dependencies.** Only `psutil` and `rich`.

## Supported hardware

| Type | Detection |
| --- | --- |
| NVIDIA GPUs | ✅ Automatic |
| Intel / AMD integrated (Windows) | ✅ Automatic (uses shared RAM) |
| Apple Silicon | ✅ Automatic (uses unified memory) |
| AMD / Intel dedicated GPUs | ⚠️ Name detected, VRAM from database |
| Linux AMD / Intel | 🚧 Not yet |

## Contributing

- **Add a model:** add a line to `data/models.json` (`params_b`, plus `active_b` for MoE models).
- **Add a GPU:** add its VRAM and memory bandwidth to `data/gpus.json`.
- **Share real benchmarks:** open an issue with your GPU, model and actual tokens/sec. It helps make the estimates more accurate.

Run `python test_estimate.py` before opening a pull request.

## License

MIT