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
from retrieval import load_retriever, retrieve_context
from sarvam_llm import sarvam_chat, SarvamError

st.set_page_config(page_title="Bike Troubleshooting Assistant",
                   page_icon="🏍️")

# Shows the retrieved manual excerpts under each answer. Useful while testing,
# but OFF for delivery so the interviewer sees a clean interface.
SHOW_RETRIEVAL_DEBUG = False

# ---------------------------------------------------------------------------
# The grounding + guardrail prompt.
#
# It opens and closes with a hard NO-THINKING directive: sarvam-30b is a
# reasoning model and will narrate its deliberation if allowed. The prompt
# describes the OUTPUT only — no procedural verbs ("analyze", "scan"), no
# numbered checklist — so there is no process for the model to narrate.
#
# The anti-stitching rule is critical for grounding: if a specific value is
# missing from the excerpts, the model must say so rather than assembling a
# fake spec out of unrelated nearby numbers.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """/no_think
Answer immediately and directly. Do NOT think out loud. Do NOT write any \
analysis, planning, drafts, numbered reasoning, or phrases like "Let's try", \
"Initial thought", or "Analyze the request". Output ONLY the final answer.

You are a customer-facing troubleshooting assistant for the {bike} \
motorcycle. You are talking directly to the bike's owner.

You will be given excerpts from the official {bike} owner's manual. Base your \
reply only on those excerpts and on the rules below.

- If the excerpts answer the question: give a short, practical reply in \
{reply_language} — a few sentences, or 2-4 brief steps. Plain, friendly \
language an owner can act on.
- If the excerpts do not cover the question: say in one or two sentences that \
the manual does not cover this specific issue, mention the closest related \
point if there is one, and suggest visiting an authorised {bike} service \
centre.
- If the question is not about this motorcycle (sport, news, weather, general \
chat, anything off-topic): reply with exactly one sentence saying you can \
only help with {bike} troubleshooting and maintenance. Nothing more.
- Never use outside knowledge. Never invent part names, specifications, \
torque values, or steps that are not in the excerpts.
- Quote figures, specifications, and units exactly as they appear in the \
excerpts. Do not reformat, "correct", convert, or guess units even if a \
value looks unusual, garbled, or incomplete.
- If a specific value (such as a tyre pressure, torque, capacity, or gap) is \
not clearly and completely stated in the excerpts, do NOT assemble one from \
unrelated numbers found nearby. Instead, say the manual does not list that \
specific value and suggest checking with an authorised {bike} service centre. \
A wrong number is worse than no number.

Write ONLY the final reply the owner should see — no analysis, no planning, \
no step numbers, no notes about the manual. Speak directly, as if you simply \
know the answer."""

# ---------------------------------------------------------------------------
# Language preprocessing — one Sarvam call that handles English, Devanagari
# Hindi, AND Hinglish (Hindi typed in Roman letters).
# ---------------------------------------------------------------------------
PREPROCESS_PROMPT = """/no_think
Answer directly with only the JSON described below — no analysis, no thinking.

You are a language preprocessor for a motorcycle troubleshooting assistant.

The user asked: "{question}"

Do two things:
1. Decide the reply language. Use "hindi" if the question is in Hindi —
   whether written in Devanagari script (for example "bike start nahi") OR in
   Roman / Hinglish style (for example "bike start nahi ho rahi" or "engine se
   awaaz aa rahi hai"). Use "english" only if it is plain English.
2. Write a clean, plain-English version of the question, suitable for
   searching an English manual.

Reply with ONLY a JSON object and nothing else, exactly in this form:
{{"language": "english" or "hindi", "english_query": "..."}}"""


def preprocess(question: str):
    """Return (reply_language, english_query).

    reply_language is "English" or "Hindi"; english_query is a clean English
    query used for manual search. Falls back safely if anything goes wrong.
    """
    try:
        raw = sarvam_chat(
            [{"role": "user",
              "content": PREPROCESS_PROMPT.format(question=question)}],
            temperature=0.0, max_tokens=1500)
        # Pull the JSON object out even if wrapped in code fences or text.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0))
        language = str(data.get("language", "english")).strip().lower()
        english_query = (data.get("english_query") or question).strip()
    except (SarvamError, json.JSONDecodeError, AttributeError,
            KeyError, TypeError):
        # Safe fallback: Devanagari script -> Hindi, otherwise English.
        language = ("hindi" if re.search(r"[\u0900-\u097F]", question)
                    else "english")
        english_query = question

    reply_language = "Hindi" if language == "hindi" else "English"
    return reply_language, english_query


@st.cache_resource(show_spinner=False)
def get_retriever(bike_key: str):
    """Load and cache one bike's hybrid retriever (built once per session)."""
    return load_retriever(bike_key)


def answer_question(bike_key: str, bike_name: str, question: str) -> str:
    """Retrieve from the manual and generate a grounded answer."""
    retriever = get_retriever(bike_key)

    reply_language, search_query = preprocess(question)
    context = retrieve_context(retriever, search_query)

    if SHOW_RETRIEVAL_DEBUG:
        with st.expander("🔎 Manual excerpts used (debug)"):
            st.caption(f"Searched the {bike_name} index with: "
                       f"\"{search_query}\"")
            st.text(context if context.strip() else "(no excerpts retrieved)")

    messages = [
        {"role": "system",
         "content": SYSTEM_PROMPT.format(bike=bike_name,
                                         reply_language=reply_language)},
        {"role": "user", "content":
            f"Manual excerpts for the {bike_name}:\n"
            f"---\n{context}\n---\n\n"
            f"User question: {question}"},
    ]
    # 4000 is the answer ceiling (the Sarvam starter tier caps max_tokens at
    # 4096). TOP_K is kept small in config.py so the prompt stays well within.
    return sarvam_chat(messages, temperature=0.0, max_tokens=4000)


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

# Handle a new question.
question = st.chat_input(f"Ask about your {bike_name}...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking the manual..."):
            try:
                reply = answer_question(bike_key, bike_name, question)
            except FileNotFoundError as exc:
                reply = (f"⚠️ {exc}\n\nThe search index for this bike is "
                         f"missing. Run `python ingest.py` to build it.")
            except SarvamError as exc:
                reply = f"⚠️ Could not reach the Sarvam API.\n\n{exc}"
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
