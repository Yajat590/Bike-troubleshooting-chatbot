# Bike Troubleshooting Assistant

A grounded AI assistant that answers motorcycle troubleshooting and maintenance
questions using **only** the official owner's manual of the selected bike — in
English, Hindi, or Hinglish. It is built so that it will not invent answers,
will not use the open web, and will politely refuse anything off-topic.

> **Live demo:** <ADD-YOUR-STREAMLIT-URL-HERE>  *(or: "Run locally — see Setup below.")*

---

## What it does

A user picks one of five popular Indian motorcycles, types a problem in plain
language — for example *"bike not starting"*, *"smoke from the exhaust"*, or
*"bike start nahi ho rahi"* — and gets a short, practical answer drawn strictly
from that bike's owner's manual.

Supported bikes: Hero Splendor Plus, Honda Shine, Bajaj Pulsar 150,
TVS Apache RTR 160 4V, Royal Enfield Classic 350.

Three behaviours define the product:

- **Grounded** — every answer comes from the manual; the model is given the
  relevant manual excerpts and instructed to use nothing else.
- **Honest** — if the manual does not cover something, it says so and points
  to an authorised service centre, rather than guessing.
- **Guarded** — off-topic questions (sports, news, chit-chat) are refused in
  one line.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Sarvam API key
#    Copy .env.example to .env and paste your key:
#    SARVAM_API_KEY=sk_your_key_here

# 3. Build the per-bike search indexes from the manuals
python ingest.py

# 4. Run the app
streamlit run app.py
```

A free Sarvam API key is available at https://dashboard.sarvam.ai. The
embedding model runs locally and downloads automatically on first run
(~80 MB, one time).

---

## How it works

Each question flows through four steps:

1. **Language detection** — a lightweight model call detects whether the
   question is English, Hindi (Devanagari), or Hinglish (Hindi in Roman
   script), and produces a clean English version of the query for searching.
2. **Retrieval** — the English query runs a hybrid search over the selected
   bike's manual and pulls the most relevant excerpts.
3. **Grounded generation** — the excerpts plus a strict system prompt are sent
   to the Sarvam LLM, which writes an answer using only that material.
4. **Language-matched reply** — the answer is returned in the user's original
   language (Devanagari Hindi for Hindi/Hinglish questions).

**Tech stack:** Python, Streamlit (UI), LlamaIndex (retrieval), FAISS-backed
vector index + BM25 (hybrid search), local `sentence-transformers` embeddings,
and the Sarvam Chat Completion API for generation.

---

## Design decisions

The task asked for the reasoning behind each choice, not just the choices.

### 1. Pre-loaded manuals for 5 popular bikes — not user-uploaded manuals

The brief allowed either a built-in knowledge base of common bikes or letting
the user upload their own manual. **I chose a curated, pre-loaded set of five
high-market-share bikes.**

*Reasoning:* grounding quality depends entirely on the quality of the source
text. Curated manuals are verified, consistent, and indexed ahead of time, so
the assistant behaves predictably and the demo works instantly with zero setup
for the evaluator. User-uploaded PDFs vary enormously in quality — scans,
broken layouts, password protection — which would make retrieval unreliable
and failures hard to diagnose. Choosing the five top-selling bikes also keeps
the assistant relevant to the largest share of Indian riders. In-chat upload
is a natural v2 extension once an upload-validation and ingestion pipeline is
in place.

### 2. Sarvam LLM, with reasoning disabled

The task allowed any model; I used Sarvam, which fits the spirit of the brief
and is strong on Indian languages — important for the Hindi/Hinglish support.

`sarvam-30b` is a reasoning model. For grounded retrieval QA, chain-of-thought
adds no accuracy — the answer is already contained in the supplied excerpts —
but it adds latency, token cost, and a tendency for internal "thinking" to
leak into the reply. I therefore **disabled reasoning** (`reasoning_effort` set
to `None`). I also evaluated the larger `sarvam-105b`; for this task the extra
model size gave no meaningful gain, because output quality here is bounded by
*retrieval* quality, not model intelligence — so the lighter, faster model was
the better engineering choice.

### 3. Hybrid retrieval — semantic + keyword

Search combines vector (semantic) search with BM25 (keyword) search, fused by
reciprocal-rank. *Reasoning:* semantic search understands intent (*"won't
start"* matching a section on ignition), while BM25 reliably catches exact
terms — part names, model-specific words — that embeddings can miss. Together
they retrieve more relevant excerpts than either alone.

### 4. Strict guardrails in the system prompt

The system prompt enforces: answer only from the excerpts; never use outside
knowledge; refuse off-topic questions in one sentence; quote figures and units
exactly as written; and — importantly — never assemble a specification (e.g. a
tyre pressure) out of unrelated numbers found nearby. A wrong number is worse
than an honest "not listed."

### 5. Multilingual handling via a preprocessing step

Rather than maintaining manuals in multiple languages, the system translates
the *query* to English for search, then answers in the user's language. This
keeps a single English knowledge base while still serving Hindi and Hinglish
users — the way people actually type in India.

---

## Known limitations

Stated honestly, because understanding a system's limits is part of building
it well:

- **PDF extraction is imperfect.** Owner's manuals contain tables and
  multi-column layouts. Some values — notably exact tyre pressures, which are
  stored in image-based tables — do not survive text extraction. When this
  happens the assistant correctly reports that the value is not listed and
  refers the user to a service centre, rather than guessing. This is an
  inherent limitation of text-based retrieval over real-world PDFs.
- **Retrieval quality varies by question.** Because chunk quality varies with
  the source layout, some questions retrieve cleaner context than others.
- **Scope is the five included manuals.** Other bikes are out of scope by
  design (see Design Decision 1).

---

## Why this matters (GTM perspective)

Owner's manuals hold the answer to most everyday rider questions, but almost
nobody reads them — the information is locked in a 100-page PDF. This assistant
turns that PDF into a conversation, in the language the rider actually speaks.

Realistic users: **dealership and OEM customer-support desks** handling repeat
"why won't it start" calls; **roadside-assistance** services needing fast,
manual-accurate guidance; and **riders** themselves via a self-service channel.

The grounding discipline is the core of the value proposition. For a vehicle,
a confidently wrong answer is a safety and liability problem — so an assistant
that stays strictly inside the official manual, and openly says when it does
not know, is far more trustworthy than a general chatbot. That trust is what
makes it deployable for a brand.

---

## Project structure

```
config.py        Central settings (bikes, chunking, retrieval, model)
ingest.py        Builds the per-bike search indexes from the manual PDFs
retrieval.py     Hybrid (semantic + keyword) retriever
sarvam_llm.py    Sarvam Chat Completion API wrapper
app.py           Streamlit chat application
manuals/         The five owner's-manual PDFs
.env.example     Template for the API key
```
