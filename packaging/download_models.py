"""Pre-download Hugging Face + NLTK models into build/models_cache/.

Run once before building the PyInstaller bundle so the AppImage works offline.
Re-runs are fast (HuggingFace / NLTK skip already-cached resources).

Usage:
    python packaging/download_models.py
"""

import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "build" / "models_cache"
HF_CACHE = CACHE_DIR / "hf"
NLTK_CACHE = CACHE_DIR / "nltk_data"

HF_CACHE.mkdir(parents=True, exist_ok=True)
NLTK_CACHE.mkdir(parents=True, exist_ok=True)

# Direct the HF library to download into our cache directory.
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE / "hub")

print(f"==> Cache root: {CACHE_DIR}")

GRAMMAR_MODEL = "prithivida/grammar_error_correcter_v1"

# --- HuggingFace ---
from transformers import (GPT2TokenizerFast, GPT2LMHeadModel,
                          AutoTokenizer, AutoModelForSeq2SeqLM)

print("==> distilgpt2 (tokenizer + LM head)")
GPT2TokenizerFast.from_pretrained("distilgpt2")
GPT2LMHeadModel.from_pretrained("distilgpt2")

print(f"==> {GRAMMAR_MODEL} (tokenizer + seq2seq)")
AutoTokenizer.from_pretrained(GRAMMAR_MODEL)
AutoModelForSeq2SeqLM.from_pretrained(GRAMMAR_MODEL)

# --- NLTK ---
import nltk
nltk.data.path.insert(0, str(NLTK_CACHE))

print("==> NLTK wordnet + omw-1.4")
nltk.download("wordnet", download_dir=str(NLTK_CACHE), quiet=True)
nltk.download("omw-1.4", download_dir=str(NLTK_CACHE), quiet=True)

total = sum(f.stat().st_size for f in CACHE_DIR.rglob("*") if f.is_file())
print(f"==> Done. Cache size: {total / 1024**2:.0f} MB")
