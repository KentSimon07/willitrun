import platform, subprocess, psutil

def get_ram_gb():
    return round(psutil.virtual_memory().total / 1024**3, 1)

def get_gpu():
    # NVIDIA
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip().split("\n")[0]
        name, vram = out.split(", ")
        return {"name": name, "vram_gb": round(int(vram) / 1024, 1), "type": "nvidia"}
    except Exception:
        pass

    # Apple Silicon (uses shared RAM)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return {"name": "Apple Silicon", "vram_gb": round(get_ram_gb() * 0.75, 1), "type": "apple"}

    # Windows: Intel / AMD
    if platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController).Name"],
                text=True,
            )
            names = [n.strip() for n in out.splitlines() if n.strip()]
            if names:
                name = names[0]
                dedicated = "Radeon RX" in name or "Arc(TM) A" in name
                if dedicated:
                    # Windows can't report dedicated VRAM reliably; ask the user later
                    return {"name": name, "vram_gb": None, "type": "dedicated"}
                # Integrated: shares about half of system RAM
                return {"name": name, "vram_gb": round(get_ram_gb() * 0.5, 1), "type": "integrated"}
        except Exception:
            pass

    return None

if __name__ == "__main__":
    print("RAM:", get_ram_gb(), "GB")
    print("GPU:", get_gpu() or "Not detected")