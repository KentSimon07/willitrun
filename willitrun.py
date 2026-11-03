from rich.console import Console
from rich.table import Table

from detect import get_gpu, get_ram_gb
from estimate import estimate_all

ICONS = {
    "fits": "[green]✅ Fits[/]",
    "partial": "[yellow]⚠️ Partial[/]",
    "cpu": "[yellow]🐢 CPU only[/]",
    "no": "[red]❌ Too big[/]",
}


def main():
    console = Console()
    ram = get_ram_gb()
    gpu = get_gpu()

    gpu_text = f"{gpu['name']} ({gpu['vram_gb']} GB, {gpu['type']})" if gpu else "Not detected"
    console.print(f"\n[bold]GPU:[/] {gpu_text}   [bold]RAM:[/] {ram} GB\n")

    results = estimate_all(gpu, ram)

    table = Table(title="Will it run?")
    table.add_column("Model")
    table.add_column("Quant")
    table.add_column("Needs", justify="right")
    table.add_column("Status")
    table.add_column("Speed", justify="right")
    table.add_column("Run with")

    for r in results:
        speed = f"~{r['tok_s']} tok/s" if r["tok_s"] else "-"
        table.add_row(r["name"], r["quant"], f"{r['need_gb']} GB",
                      ICONS[r["status"]], speed, f"ollama run {r['ollama']}")

    console.print(table)

    runnable = [r for r in results if r["status"] != "no" and r["tok_s"] >= 10]
    if runnable:
        best = max(runnable, key=lambda r: (r["need_gb"], r["tok_s"]))
        console.print(f"\n[bold green]Best for you:[/] {best['name']} {best['quant']} "
                      f"(~{best['tok_s']} tok/s) → ollama run {best['ollama']}\n")


if __name__ == "__main__":
    main()