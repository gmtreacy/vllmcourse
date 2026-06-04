# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "llmcompressor==0.7.1",
#   "vllm==0.11.0",
#   "transformers>=4.47.0",
#   "datasets>=2.21.0",
#   "torch>=2.4.0",
# ]
# ///

import os
import pathlib
import warnings

import torch
from huggingface_hub import snapshot_download
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_ID   = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "./Qwen3-0.6B-W4A16"

# ── sanity check ────────────────────────────────────────────────────────────
print(f"torch:  {torch.__version__}")
print(f"cuda:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"vram:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

# ── recipe ───────────────────────────────────────────────────────────────────
recipe = GPTQModifier(
    scheme="W4A16",
    targets="Linear",
    ignore=["lm_head"],
)
print(f"Recipe: {recipe}\n")

# ── quantize ─────────────────────────────────────────────────────────────────
if os.path.isdir(OUTPUT_DIR):
    print(f"Output dir {OUTPUT_DIR!r} already exists — skipping quantization.")
else:
    print("Starting quantization …")
    oneshot(
        model=MODEL_ID,
        dataset="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        recipe=recipe,  # type: ignore[arg-type]
        output_dir=OUTPUT_DIR,
        max_seq_length=4096,
        num_calibration_samples=256,
    )
    print(f"\nQuantization complete. Model saved to: {OUTPUT_DIR}")

# ── size comparison ───────────────────────────────────────────────────────────
def folder_size(path):
    p = pathlib.Path(path)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

def fmt(nbytes):
    if nbytes < 1024 ** 2:
        return f"{nbytes/1024:.1f} KB"
    if nbytes < 1024 ** 3:
        return f"{nbytes/1024**2:.1f} MB"
    return f"{nbytes/1024**3:.2f} GB"

print("\nResolving original model cache path …")
model_cache = snapshot_download(MODEL_ID)

size_orig = folder_size(model_cache)
size_q    = folder_size(OUTPUT_DIR)
reduction = (1 - size_q / size_orig) * 100 if size_orig > 0 else 0

print("\nModel Size Comparison")
print("=" * 45)
print(f"Original (BF16):    {fmt(size_orig)}")
print(f"Quantized (W4A16):  {fmt(size_q)}")
print(f"Reduction:          {reduction:.0f}%")
