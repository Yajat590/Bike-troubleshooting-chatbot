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
load_dotenv()

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
TOP_K = 5               # how many whole chunks hybrid search feeds to the LLM

# --- Embeddings (runs locally — free, no API key, nothing to rate-limit) ---
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Sarvam LLM ------------------------------------------------------------
SARVAM_API_URL = "https://api.sarvam.ai/v1/chat/completions"
SARVAM_MODEL = "sarvam-30b"            # 64K context, free per token
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip()

# Safety guard for the contextualisation step: caps the document text sent
# with each chunk. All five manuals fit well inside sarvam-30b's 64K window,
# so this only prevents a freakishly large file from erroring out.
MAX_DOC_CHARS = 240_000
