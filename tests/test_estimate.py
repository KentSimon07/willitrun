from estimate import estimate, load_json

DB = load_json("gpus.json")
NVIDIA_8GB = {"name": "NVIDIA GeForce RTX 4060", "vram_gb": 8, "type": "nvidia"}
IGPU = {"name": "Intel(R) Arc(TM) 140T GPU", "vram_gb": 15.7, "type": "integrated"}

small = {"name": "S", "ollama": "s", "params_b": 3}
big = {"name": "B", "ollama": "b", "params_b": 70}
moe = {"name": "M", "ollama": "m", "params_b": 30, "active_b": 3}
dense30 = {"name": "D", "ollama": "d", "params_b": 30}

assert estimate(small, NVIDIA_8GB, 16, DB)["status"] == "fits"
assert estimate(big, NVIDIA_8GB, 16, DB)["status"] == "no"
assert estimate(dense30, IGPU, 31.4, DB)["status"] == "cpu"   # integrated spills to shared RAM
assert estimate(small, None, 16, DB)["status"] == "cpu"       # no GPU detected
# MoE runs much faster than a dense model of the same size
assert estimate(moe, IGPU, 31.4, DB)["tok_s"] > 5 * estimate(dense30, IGPU, 31.4, DB)["tok_s"]
print("all tests passed")