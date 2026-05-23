"""Streamlit chat app — the Bike Troubleshooting Assistant.

Flow for every user question:
  1. Detect the language (English / Hindi / Hinglish) and get a clean English
     version of the question for searching.
  2. Hybrid-search the selected bike's manual with the English version.
  3. Ask Sarvam to answer using ONLY those chunks, with strict guardrails.
  4. Reply in English for English questions, and in proper Hindi (Devanagari)
     for Hindi or Hinglish questions.

Run locally:   streamlit run app.py
"""
import json
import re

import streamlit as st

import config
from retrieval import load_retriever, retrieve_context, retrieve_context_multi
from sarvam_llm import sarvam_chat, sarvam_chat_stream, SarvamError
from sarvam_stt import sarvam_transcribe

st.set_page_config(page_title="Bike Troubleshooting Assistant",
                   page_icon="🏍️")

# Shows the retrieved manual excerpts under each answer. Useful while testing,
# but OFF for delivery so the interviewer sees a clean interface.
SHOW_RETRIEVAL_DEBUG = False


# ---------------------------------------------------------------------------
# Eager startup: load embedding model + all bike indexes once on app boot.
# Cached across reruns — only the very first page load pays the cost.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Starting up — loading search engine...")
def _preload_retrievers():
    from retrieval import get_embed_model
    get_embed_model()
    retrievers = {}
    for bike_key in config.BIKES:
        try:
            retrievers[bike_key] = load_retriever(bike_key)
        except FileNotFoundError:
            pass
    return retrievers


_RETRIEVERS = _preload_retrievers()

# ---------------------------------------------------------------------------
# The grounding + guardrail prompt.
#
# Reasoning is disabled at the API level (reasoning_effort=None in
# sarvam_llm.py), so the prompt no longer needs a /no_think directive. It
# describes the OUTPUT only — no procedural verbs, no numbered checklist — and
# explicitly asks for brief, succinct answers.
#
# The anti-stitching rule is critical for grounding: if a specific value is
# missing from the excerpts, the model must say so rather than assembling a
# fake spec out of unrelated nearby numbers.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a friendly mechanic helping the owner of a \
{bike}. Talk like a knowledgeable friend — warm, direct, easy to follow.

You will receive excerpts from the {bike} owner's manual. Use ONLY those \
excerpts. Reply in {reply_language}.

STYLE:
- Start with what to do, not what might be wrong. Lead with the fix.
- Use numbered steps (1, 2, 3) for any procedure. Keep each step to one \
action.
- Use exact values from the excerpts (pressures, torques, gaps). Never \
round, convert, or guess.
- If the excerpts mention multiple wheel types or variants, pick the one \
that matches the {bike}. If unclear, state both clearly labelled.
- Keep it short: 2-5 steps or 2-3 sentences max.

BOUNDARIES:
- If the excerpts don't answer the question: say so in one line and suggest \
the nearest {bike} service centre.
- Off-topic questions: one sentence — you only help with {bike} maintenance.
- Never invent specs, part names, or steps not in the excerpts.
- If a specific value is missing from the excerpts, say so — don't stitch \
one together from nearby numbers."""

# ---------------------------------------------------------------------------
# Language preprocessing — one Sarvam call that handles English, Devanagari
# Hindi, AND Hinglish (Hindi typed in Roman letters).
# ---------------------------------------------------------------------------
PREPROCESS_PROMPT = """Detect language and generate search queries. \
Reply with ONLY JSON, nothing else.

User asked: "{question}"

1. language: "hindi" if Hindi/Devanagari/Hinglish, else "english".
2. english_query: clean English version for manual search.
3. alt_queries: 2 alternative search phrases using manual headings, \
synonyms, or technical terms.

{{"language":"...","english_query":"...","alt_queries":["...","..."]}}"""

RETRY_QUERY_PROMPT = """User asked about {bike}: "{question}"
Manual search found no direct answer. Suggest 3 alternative search \
phrases using different terms, manual section names, or broader/narrower \
scope. Reply with ONLY a JSON array: ["phrase1","phrase2","phrase3"]"""

# Strictly "I couldn't find this in the manual" phrases.
# IMPORTANT: do NOT add generic "visit an authorised service centre" type
# phrases here — they appear in perfectly good answers as routine follow-up
# advice and would cause the retry to fire on a complete reply, producing
# a duplicate second answer below the first.
_INSUFFICIENT_MARKERS = [
    "manual does not cover",
    "manual doesn't cover",
    "manual does not list",
    "manual doesn't list",
    "manual does not specify",
    "manual doesn't specify",
    "manual does not contain",
    "manual doesn't contain",
    "manual does not mention",
    "manual doesn't mention",
    "not mentioned in the manual",
    "not found in the manual",
    "not in the excerpts",
    "no information in the manual",
]


def _is_hindi(question: str) -> bool:
    """True if the question contains Devanagari or any Hinglish marker.

    A single marker is enough \u2014 these words are rare in English but extremely
    common in transliterated Hindi, so requiring two was causing short queries
    like "bike start nahi" to be misclassified as English. Words that overlap
    with everyday English (to, me, the, hi, on, in) are intentionally excluded.
    """
    if re.search(r"[\u0900-\u097F]", question):
        return True
    hinglish = {
        # Negation / question words
        "nahi", "nahin", "nai", "mat", "kyu", "kyun", "kya", "kaun",
        "kab", "kahan", "kaise", "kaisa", "kaisi",
        "kitna", "kitne", "kitni", "kiska", "kiski", "kiske",
        # Postpositions / particles (skipping ambiguous "to", "me", "hi")
        "ka", "ki", "ke", "ko", "se", "par", "tak", "mein", "bhi",
        # "to be" forms — "the" is intentionally OMITTED. It is the Hindi
        # past-tense plural ("they were"), but it is also the single most
        # common word in English, so including it would misclassify almost
        # every English query as Hindi.
        "hai", "hain", "tha", "thi", "ho", "hoga", "hogi", "honge",
        # Continuous-tense helpers
        "raha", "rahi", "rahe", "rha", "rhi", "rhe",
        # Common verbs
        "karna", "karta", "karte", "karti", "kar", "kiya", "kare",
        "hota", "hoti", "hote", "hua", "hui",
        "aa", "aata", "aati", "aate", "aaya", "aayi", "aana",
        "jana", "jaata", "jaati", "jaate", "gaya", "gayi",
        "lagta", "lagti", "lagte", "laga", "lagi",
        "chahiye", "chahta", "chahti",
        "badalna", "badalta", "badalti", "badal",
        "chal", "chalu", "chalna", "chalta", "chalti", "chalti",
        "band", "chalu", "ruk", "rukna",
        "dena", "deta", "deti", "diya", "dijiye",
        "milna", "milta", "milti", "mila", "mili",
        # Adjectives / descriptors
        "thik", "theek", "sahi", "thoda", "thodi", "thode",
        "puri", "pura", "khatam",
        # Possessives
        "mera", "meri", "mere", "tera", "teri", "tere",
        # Descriptive / problem words
        "awaaz", "awaz", "dhuan", "dhua", "dikkat", "garbar",
        "garam", "thanda", "tezz", "tez", "dheere", "dhire",
        "wala", "wali", "wale",
    }
    words = set(re.findall(r"\b[a-z]+\b", question.lower()))
    return bool(words & hinglish)


def preprocess(question: str):
    """Return (reply_language, search_queries).

    For plain English questions, skips the LLM call entirely and builds
    search queries locally \u2014 saving ~2-3 seconds.
    """
    is_hindi = _is_hindi(question)

    if not is_hindi:
        return "English", [question]

    try:
        raw = sarvam_chat(
            [{"role": "user",
              "content": PREPROCESS_PROMPT.format(question=question)}],
            temperature=0.0, max_tokens=300)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0))
        english_query = (data.get("english_query") or question).strip()
        alt_queries = data.get("alt_queries", [])
        if not isinstance(alt_queries, list):
            alt_queries = []
        alt_queries = [q.strip() for q in alt_queries
                       if isinstance(q, str) and q.strip()]
    except (SarvamError, json.JSONDecodeError, AttributeError,
            KeyError, TypeError):
        english_query = question
        alt_queries = []

    search_queries = [english_query] + alt_queries[:2]
    return "Hindi", search_queries


def get_retriever(bike_key: str):
    """Look up the preloaded retriever, or load on demand as fallback."""
    if bike_key in _RETRIEVERS:
        return _RETRIEVERS[bike_key]
    retriever = load_retriever(bike_key)
    _RETRIEVERS[bike_key] = retriever
    return retriever


def _answer_seems_insufficient(answer: str) -> bool:
    """True if the answer indicates the manual didn't have enough information.

    Two guards keep this strict:
      1. Length cap — the system prompt instructs short (one-line) "couldn't
         find" replies. Anything longer than ~250 chars is a real, substantive
         answer that happens to mention a service centre as routine advice.
      2. Marker list — only unambiguous "manual does/doesn't ..." phrases.
    Without these guards, the retry path fires on complete answers and
    appends a duplicate second answer below the first.
    """
    if len(answer) > 250:
        return False
    low = answer.lower()
    return any(m in low for m in _INSUFFICIENT_MARKERS)


def _generate_retry_queries(bike_name: str, question: str):
    """Ask Sarvam for alternative search phrases when the first attempt missed."""
    try:
        raw = sarvam_chat(
            [{"role": "user",
              "content": RETRY_QUERY_PROMPT.format(bike=bike_name,
                                                    question=question)}],
            temperature=0.3, max_tokens=500)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        queries = json.loads(match.group(0))
        return [q.strip() for q in queries
                if isinstance(q, str) and q.strip()][:3]
    except (SarvamError, json.JSONDecodeError, AttributeError, TypeError):
        return []


def _build_messages(bike_name, reply_language, context, question):
    """Build the LLM messages list."""
    if len(context) > config.MAX_CONTEXT_CHARS:
        context = context[:config.MAX_CONTEXT_CHARS]
    return [
        {"role": "system",
         "content": SYSTEM_PROMPT.format(bike=bike_name,
                                         reply_language=reply_language)},
        {"role": "user", "content":
            f"Manual excerpts for the {bike_name}:\n"
            f"---\n{context}\n---\n\n"
            f"User question: {question}"},
    ]


def answer_question(bike_key: str, bike_name: str, question: str) -> str:
    """Agentic retrieval: search with status, then stream the answer."""

    # --- Steps 1-2: search (shown in status widget) ---
    with st.status("Searching...", expanded=True, state="running") as status:

        status.update(label="Step 1/2 — Understanding your question...",
                      state="running")
        reply_language, search_queries = preprocess(question)
        if reply_language == "Hindi":
            status.write("Detected Hindi/Hinglish — translated for search.")
        status.write(f"Search queries: *{', '.join(search_queries)}*")

        status.update(label="Step 2/2 — Searching the manual...",
                      state="running")
        retriever = get_retriever(bike_key)
        context = retrieve_context_multi(retriever, search_queries)
        chunk_count = context.count("---") + 1 if context.strip() else 0
        status.write(f"Found {chunk_count} relevant sections.")

        if SHOW_RETRIEVAL_DEBUG:
            with st.expander("Manual excerpts (debug)"):
                st.caption(f"Searched with: {search_queries}")
                st.text(context if context.strip()
                        else "(no excerpts retrieved)")

        status.update(label="Search complete — streaming answer...",
                      expanded=False, state="complete")

    # --- Step 3: stream the answer directly into the chat bubble ---
    messages = _build_messages(bike_name, reply_language, context, question)
    reply = st.write_stream(sarvam_chat_stream(messages, max_tokens=500))

    # --- Auto-retry if insufficient (non-streamed, rare with sarvam-m) ---
    if _answer_seems_insufficient(reply):
        retry_queries = _generate_retry_queries(bike_name, question)
        if retry_queries:
            extra_context = retrieve_context_multi(retriever, retry_queries)
            if extra_context.strip():
                combined = context + "\n\n---\n\n" + extra_context
                msgs = _build_messages(bike_name, reply_language,
                                       combined, question)
                retry_reply = sarvam_chat(msgs, temperature=0.0,
                                          max_tokens=500)
                if not _answer_seems_insufficient(retry_reply):
                    st.markdown("---")
                    st.markdown(retry_reply)
                    reply = retry_reply

    return reply


# ---------------------------------------------------------------------------
# Sidebar — bike selector
# ---------------------------------------------------------------------------
st.sidebar.title("🏍️ Bike Assistant")
bike_name = st.sidebar.selectbox("Select your motorcycle",
                                 list(config.BIKES.values()))
bike_key = {v: k for k, v in config.BIKES.items()}[bike_name]
st.sidebar.caption(
    "Answers are grounded only in this bike's official owner's manual. "
    "Ask in English, Hindi, or Hinglish.")
if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Bike Troubleshooting Assistant")
st.caption(f"Currently helping with: **{bike_name}**")

# Reset the conversation whenever the user switches bikes.
if st.session_state.get("bike_key") != bike_key:
    st.session_state.bike_key = bike_key
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info("Describe a problem with your bike — for example "
            "*\"bike not starting\"*, *\"front brake feels loose\"*, "
            "or *\"engine oil kitne din me badalna chahiye\"*.")

# Replay the conversation so far.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat input with a built-in mic icon (Streamlit 1.56+).
#
# accept_audio=True puts a small recording button INSIDE the chat input
# box — the same in-input voice UX ChatGPT / Claude / Gemini show. The
# returned `prompt` is a dict-like object with two fields:
#   prompt.text   — the typed message ("" if the user only recorded audio)
#   prompt.audio  — an UploadedFile (WAV) if the user recorded, else None
# We prefer typed text when both are provided; otherwise we send the WAV
# bytes through Sarvam Speech-to-Text and treat the transcript exactly
# like typed input. audio_sample_rate defaults to 16000 Hz — optimal for
# speech recognition, as the Streamlit docs note.
# ---------------------------------------------------------------------------
# audio_sample_rate=24000 — a step above the 16 kHz default. Browser
# WebRTC capture quality on repeated recordings can drift at 16 kHz
# (auto-gain / echo cancel re-adapting); 24 kHz gives the STT model a
# cleaner signal at a tiny file-size cost.
prompt = st.chat_input(f"Ask about your {bike_name}...",
                       accept_audio=True,
                       audio_sample_rate=24000)

question = None
if prompt:
    if prompt.text:
        question = prompt.text
    elif prompt.audio is not None:
        with st.spinner("Transcribing your voice..."):
            try:
                question = sarvam_transcribe(prompt.audio.getvalue())
            except SarvamError as exc:
                st.error(str(exc))

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            reply = answer_question(bike_key, bike_name, question)
        except FileNotFoundError as exc:
            reply = (f"⚠️ {exc}\n\nThe search index for this bike is "
                     f"missing. Run `python ingest.py` to build it.")
            st.markdown(reply)
        except SarvamError as exc:
            reply = f"⚠️ Could not reach the Sarvam API.\n\n{exc}"
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
