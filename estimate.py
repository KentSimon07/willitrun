import json
from pathlib import Path

DATA = Path(__file__).parent / "data"

BITS = {"Q8": 8.5, "Q4": 4.8}   # bits per weight, best quality first
OVERHEAD = 1.15                 # context cache + runtime
EFFICIENCY = 0.6                # real speed vs theoretical max


def load_json(name):
    with open(DATA / name, encoding="utf-8") as f:
        return json.load(f)


def lookup_gpu(gpu, gpu_db):
    """Return (bandwidth, vram_gb) for the detected GPU."""
    defaults = gpu_db["defaults"]
    if not gpu:
        return defaults["system_ram"], 0

    name = gpu["name"]
    matches = [k for k in gpu_db["gpus"] if k.lower() in name.lower()]
    if matches:
        best = gpu_db["gpus"][max(matches, key=len)]
        vram = gpu["vram_gb"] or best["vram_gb"]
        if gpu["type"] in ("integrated", "apple"):
            vram = gpu["vram_gb"]
        return best["bandwidth"], vram

    return defaults.get(gpu["type"], defaults["dedicated"]), gpu["vram_gb"] or 0


def model_size_gb(model, quant):
    if "size_gb" in model:          # fixed-size formats (e.g. ternary)
        return model["size_gb"]
    return model["params_b"] * BITS[quant] / 8


def active_size_gb(model, quant):
    """Size of the weights read per token (smaller for MoE models)."""
    size = model_size_gb(model, quant)
    if "active_b" in model:
        size *= model["active_b"] / model["params_b"]
    return size


def estimate(model, gpu, ram_gb, gpu_db):
    bandwidth, vram = lookup_gpu(gpu, gpu_db)
    ram_bw = gpu_db["defaults"]["system_ram"]
    shared = gpu is None or gpu["type"] in ("integrated", "apple")
    quants = ["Q4"] if "size_gb" in model else list(BITS)

    for quant in quants:
        need = model_size_gb(model, quant) * OVERHEAD
        active = active_size_gb(model, quant)

        # Fits fully in GPU memory
        if gpu and vram and need <= vram:
            return _result(model, quant, need, "fits", bandwidth / active)

        # Dedicated GPU + spill into system RAM
        if gpu and not shared and need <= vram + ram_gb * 0.6:
            return _result(model, quant, need, "partial", ram_bw / active)

        # CPU only, or integrated GPU spilling into the rest of shared RAM
        if shared and need <= ram_gb * 0.7:
            return _result(model, quant, need, "cpu", ram_bw / active)

    need = model_size_gb(model, "Q4") * OVERHEAD
    return _result(model, "Q4", need, "no", 0)


def _result(model, quant, need, status, tok_s):
    return {
        "name": model["name"],
        "ollama": model["ollama"],
        "quant": quant,
        "need_gb": round(need, 1),
        "status": status,
        "tok_s": round(tok_s * EFFICIENCY),
    }


def estimate_all(gpu, ram_gb):
    models = load_json("models.json")
    gpu_db = load_json("gpus.json")
    results = [estimate(m, gpu, ram_gb, gpu_db) for m in models]
    order = {"fits": 0, "partial": 1, "cpu": 2, "no": 3}
    return sorted(results, key=lambda r: (order[r["status"]], -r["tok_s"]))