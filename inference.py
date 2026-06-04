# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "llmcompressor==0.7.1",
#   "transformers>=4.47.0",
#   "datasets>=2.21.0",
#   "torch>=2.4.0",
# ]
# ///

import logging
import math
import warnings

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")
logging.getLogger("llmcompressor").setLevel(logging.WARNING)

MODEL_DIR  = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "./Qwen3-0.6B-W4A16"
PROMPT     = "Machine learning is a branch of"

# ── load tokenizer once ───────────────────────────────────────────────────────
print("Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# ── generation helper ─────────────────────────────────────────────────────────
def generate(model, label):
    inputs  = tokenizer(PROMPT, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    print(f"\n{label}")
    print(f"Prompt:   {PROMPT}")
    print(f"Response: {tokenizer.decode(generated, skip_special_tokens=True)}")

# ── perplexity helper ─────────────────────────────────────────────────────────
def calculate_perplexity(model, dataset, max_tokens=5000, stride=512):
    encodings = tokenizer(
        "\n\n".join(dataset["text"]),
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )
    input_ids          = encodings.input_ids
    nlls, prev_end     = [], 0
    for begin_loc in range(0, input_ids.size(1), stride):
        end_loc    = min(begin_loc + stride, input_ids.size(1))
        trg_len    = end_loc - prev_end
        inp        = input_ids[:, begin_loc:end_loc]
        tgt        = inp.clone()
        tgt[:, :-trg_len] = -100
        with torch.no_grad():
            loss = model(inp, labels=tgt).loss
            nlls.append(loss * trg_len)
        prev_end = end_loc
    return math.exp(torch.stack(nlls).sum() / prev_end)

# ── load test data once ───────────────────────────────────────────────────────
print("Loading wikitext-2 test split …")
test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
print(f"Loaded {len(test_data)} test samples")

# ── base model ────────────────────────────────────────────────────────────────
print("\nLoading base model …")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, device_map="cpu", torch_dtype=torch.bfloat16,
)
generate(base_model, f"Base Model ({MODEL_DIR})")
print("\nCalculating base perplexity …")
base_ppl = calculate_perplexity(base_model, test_data)
print(f"Base perplexity: {base_ppl:.2f}")

# free before loading quantized model
del base_model
torch.cuda.empty_cache()

# ── quantized model ───────────────────────────────────────────────────────────
print("\nLoading quantized model …")
quant_model = AutoModelForCausalLM.from_pretrained(
    OUTPUT_DIR, device_map="cpu", torch_dtype=torch.bfloat16,
)
generate(quant_model, f"Quantized Model ({OUTPUT_DIR})")
print("\nCalculating quantized perplexity …")
quant_ppl = calculate_perplexity(quant_model, test_data)
print(f"Quantized perplexity: {quant_ppl:.2f}")

# ── comparison ────────────────────────────────────────────────────────────────
print("\nPerplexity Comparison")
print("=" * 40)
print(f"Base (BF16):       {base_ppl:.2f}")
print(f"Quantized (W4A16): {quant_ppl:.2f}")
print(f"Difference:        {quant_ppl - base_ppl:+.2f} ({(quant_ppl/base_ppl - 1)*100:+.1f}%)")
print("\nA small increase in perplexity is expected — quantized layers use 4-bit weights.")
