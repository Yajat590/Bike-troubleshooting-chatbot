"""Central configuration for the bike troubleshooting chatbot.

Every other file imports its settings from here, so there is exactly one
place to change anything.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Loads SARVAM_API_KEY from a local .env file when running on your machine.
# On Streamlit Cloud the key comes from the dashboard's Secrets instead;
# either way it ends up in the environment, so the line below is harmless.
load_dotenv(override=True)

# --- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MANUALS_DIR = BASE_DIR / "manuals"      # the five owner's-manual PDFs go here
STORAGE_DIR = BASE_DIR / "storage"      # ingest.py writes the indexes here

# --- The knowledge base ----------------------------------------------------
# KEY   = the PDF filename prefix inside manuals/  (key "pulsar" -> pulsar.pdf)
# VALUE = the human-readable name shown in the app's bike selector.
BIKES = {
    "splendor": "Hero Splendor Plus",
    "shine":    "Honda Shine",
    "pulsar":   "Bajaj Pulsar 150",
    "apache":   "TVS Apache RTR 160 4V",
    "enfield":  "Royal Enfield Classic 350",
}

# --- Chunking --------------------------------------------------------------
CHUNK_SIZE = 512        # approx. tokens per chunk
CHUNK_OVERLAP = 64      # token overlap so sentences are not cut mid-thought

# --- Retrieval -------------------------------------------------------------
# TOP_K = 5 is a balance: enough chunks that the genuinely relevant section is
# reliably retrieved, but few enough that the prompt stays well within the
# model's context. Each chunk is passed WHOLE — the useful detail often sits
# in the middle/end of a chunk, so chunks must not be trimmed.
TOP_K = 3               # how many whole chunks hybrid search feeds to the LLM
AGENTIC_MAX_CHUNKS = 5  # max unique chunks across all agentic search queries
MAX_CONTEXT_CHARS = 5000  # hard cap on manual text sent to the LLM

# --- Embeddings (runs locally — free, no API key, nothing to rate-limit) ---
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Sarvam LLM ------------------------------------------------------------
SARVAM_API_URL = "https://api.sarvam.ai/v1/chat/completions"
SARVAM_MODEL = "sarvam-m"              # faster, 24K context, 2048 max output
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip()

# On Streamlit Cloud the key comes through st.secrets, not through the
# environment, so fall back to that if the env did not supply one. The
# import is local so non-Streamlit callers (e.g. ingest.py run from the
# CLI) don't pay the import cost or fail when no secrets.toml is present.
if not SARVAM_API_KEY:
    try:
        import streamlit as st  # type: ignore
        SARVAM_API_KEY = str(st.secrets.get("SARVAM_API_KEY", "")).strip()
    except Exception:
        # No streamlit installed, no secrets.toml, or any other failure —
        # leave the key empty. The first API call will then surface a
        # clear "SARVAM_API_KEY is not set" error rather than crashing
        # at import time.
        pass

# Safety guard for the contextualisation step: caps the document text sent
# with each chunk. All five manuals fit well inside sarvam-30b's 64K window,
# so this only prevents a freakishly large file from erroring out.
MAX_DOC_CHARS = 240_000
