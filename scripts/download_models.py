#!/usr/bin/env python3
"""Pre-download all HuggingFace models needed for the autotune experiment.

Run this on the cluster before submitting the SLURM job so that model
weights are cached locally and workers don't download at startup.

Usage:
    python scripts/download_models.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    "facebook/opt-1.3b",
    "Qwen/Qwen2-1.5B",
    "microsoft/phi-2",
    "facebook/opt-6.7b",
]


def main():
    for model_name in MODELS:
        print(f"\n{'='*50}")
        print(f"Downloading {model_name}...")
        print(f"{'='*50}")

        print(f"  Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"  Tokenizer ready (vocab size: {tokenizer.vocab_size})")

        print(f"  Model (float16)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="float16",
        )
        print(f"  Model ready ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")
        del model  # free memory before loading next

    print(f"\nAll {len(MODELS)} models downloaded and cached.")


if __name__ == "__main__":
    main()
