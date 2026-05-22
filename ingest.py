"""Ingestion pipeline — builds one contextual hybrid-search index per bike.

For every bike it:
  1. reads the owner's-manual PDF from manuals/,
  2. cleans the text and splits it into overlapping chunks,
  3. CONTEXTUAL RETRIEVAL: asks Sarvam to write a one-line context blurb for
     each chunk and prepends it (Anthropic's technique — it makes isolated
     chunks findable),
  4. builds a vector index from the contextualised chunks and saves it to
     storage/<bike>/.

The BM25 (keyword) half of hybrid search is rebuilt cheaply from these saved
chunks when the app starts, so the two halves never drift apart.

Usage:
  python ingest.py                     build every bike (with contextualising)
  python ingest.py --no-context        build fast, plain index (no Sarvam)
  python ingest.py --bikes pulsar      build a single bike
  python ingest.py --limit 5           smoke test: only 5 chunks per bike
  python ingest.py --force             rebuild even if storage/<bike>/ exists
"""
import argparse
import logging
import re
import shutil
import sys

import pypdf

# The Apache manual is a web-export PDF with a slightly irregular structure.
# pypdf recovers from it fine but prints noisy warnings — quiet them.
logging.getLogger("pypdf").setLevel(logging.ERROR)
from tqdm import tqdm
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import config
from sarvam_llm import sarvam_chat, SarvamError

# LlamaIndex wants an LLM for some helpers. We never use it (Sarvam is called
# directly), so disable it — this stops any accidental call to OpenAI.
Settings.llm = None


# ---------------------------------------------------------------------------
# 1. Read and clean the manual PDF
# ---------------------------------------------------------------------------
# Patterns for obvious web-export junk. Kept deliberately conservative so we
# never delete real manual content.
_JUNK_PATTERNS = [
    re.compile(r"^\s*\d{1,4}\s*$"),            # a line that is only a page number
    re.compile(r"manualslib", re.IGNORECASE),  # web-export site footer
    re.compile(r"^\s*https?://", re.IGNORECASE),
]


def clean_text(text: str) -> str:
    """Drop page-number-only lines and web junk; collapse big blank gaps."""
    kept = [ln for ln in text.splitlines()
            if not any(p.search(ln) for p in _JUNK_PATTERNS)]
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def read_manual(bike_key: str):
    """Return the cleaned full text of manuals/<bike_key>*.pdf, or None."""
    pdfs = sorted(config.MANUALS_DIR.glob(f"{bike_key}*.pdf"))
    if not pdfs:
        return None
    pages_text = []
    for pdf_path in pdfs:
        reader = pypdf.PdfReader(str(pdf_path))
        for page in reader.pages:
            pages_text.append(page.extract_text() or "")
    return clean_text("\n".join(pages_text))


# ---------------------------------------------------------------------------
# 2. Contextual Retrieval — generate a context blurb for a single chunk
# ---------------------------------------------------------------------------
_CONTEXT_PROMPT = """<document>
{document}
</document>

Here is a chunk taken from the above {bike} owner's manual:
<chunk>
{chunk}
</chunk>

In ONE or TWO short sentences, describe what this chunk is about and where it
sits in the manual, so a search engine can locate it. Always name the bike
model and the specific topic or section. Reply with ONLY the context
sentence(s) and nothing else."""


def contextualise(chunk_text: str, document_text: str, bike_name: str) -> str:
    """Ask Sarvam for a short context blurb. Returns '' if the call fails."""
    document = document_text[: config.MAX_DOC_CHARS]
    messages = [{
        "role": "user",
        "content": _CONTEXT_PROMPT.format(
            document=document, bike=bike_name, chunk=chunk_text),
    }]
    try:
        # Generous cap: covers the model's reasoning plus the short blurb.
        return sarvam_chat(messages, temperature=0.2, max_tokens=2000)
    except SarvamError as exc:
        # One bad call must not kill a long run — degrade gracefully.
        tqdm.write(f"    ! context skipped for one chunk ({exc})")
        return ""


# ---------------------------------------------------------------------------
# 3. Build and persist one bike's index
# ---------------------------------------------------------------------------
def build_bike(bike_key, bike_name, embed_model, use_context, limit) -> bool:
    text = read_manual(bike_key)
    if not text:
        print(f"  SKIP: no PDF found at manuals/{bike_key}*.pdf")
        return False
    print(f"  manual loaded: {len(text):,} characters")

    splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE,
                                chunk_overlap=config.CHUNK_OVERLAP)
    document = Document(text=text, metadata={"bike": bike_key})
    nodes = splitter.get_nodes_from_documents([document])
    if limit:
        nodes = nodes[:limit]
    print(f"  split into {len(nodes)} chunks")

    if use_context:
        print("  generating contextual blurbs via Sarvam...")
        for node in tqdm(nodes, desc=f"  {bike_key}", unit="chunk"):
            blurb = contextualise(node.text, text, bike_name)
            if blurb:
                # Prepend the blurb so the chunk is findable on its own.
                node.text = f"{blurb}\n\n{node.text}"

    index = VectorStoreIndex(nodes, embed_model=embed_model,
                             show_progress=False)

    persist_dir = config.STORAGE_DIR / bike_key
    if persist_dir.exists():
        shutil.rmtree(persist_dir)          # always a clean rebuild
    index.storage_context.persist(persist_dir=str(persist_dir))
    print(f"  saved index -> storage/{bike_key}/")
    return True


# ---------------------------------------------------------------------------
# 4. Command-line entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build per-bike contextual hybrid-search indexes.")
    parser.add_argument("--bikes", default="",
                        help="comma-separated bike keys (default: all)")
    parser.add_argument("--limit", type=int, default=0,
                        help="max chunks per bike, for quick tests (0 = all)")
    parser.add_argument("--no-context", action="store_true",
                        help="skip Sarvam contextualisation (fast plain index)")
    parser.add_argument("--force", action="store_true",
                        help="rebuild a bike even if its index already exists")
    args = parser.parse_args()

    selected = [b.strip() for b in args.bikes.split(",") if b.strip()] \
        or list(config.BIKES)
    unknown = [b for b in selected if b not in config.BIKES]
    if unknown:
        sys.exit(f"Unknown bike key(s): {unknown}. "
                 f"Valid keys: {list(config.BIKES)}")

    use_context = not args.no_context
    if use_context and not config.SARVAM_API_KEY:
        sys.exit("SARVAM_API_KEY is not set. Add it to .env, "
                 "or run with --no-context to build plain indexes.")

    print(f"Loading embedding model ({config.EMBED_MODEL})...")
    print("(first run downloads ~80 MB — this is a one-time download)")
    embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL)

    config.STORAGE_DIR.mkdir(exist_ok=True)
    built = 0
    for bike_key in selected:
        bike_name = config.BIKES[bike_key]
        persist_dir = config.STORAGE_DIR / bike_key
        print(f"\n=== {bike_name}  ({bike_key}) ===")
        if persist_dir.exists() and not args.force:
            print(f"  already built. Use --force to rebuild.")
            continue
        if build_bike(bike_key, bike_name, embed_model,
                       use_context, args.limit):
            built += 1

    print(f"\nDone — {built} bike index(es) built. Everything is in storage/.")
    if not use_context:
        print("Note: built WITHOUT contextual retrieval. "
              "Re-run without --no-context for the final version.")


if __name__ == "__main__":
    main()
