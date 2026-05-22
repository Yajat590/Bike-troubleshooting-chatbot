"""Thin wrapper around the Sarvam Chat Completion API.

We call Sarvam directly with `requests` rather than through a framework's LLM
class. The contract is then completely explicit and there is nothing hidden
that can silently break.

API reference: https://docs.sarvam.ai/api-reference-docs/chat/chat-completions
"""
import re
import time

import requests

from config import SARVAM_API_URL, SARVAM_MODEL, SARVAM_API_KEY


class SarvamError(RuntimeError):
    """Raised when the Sarvam API cannot be reached or returns an error."""


# Phrases that only appear when sarvam-30b is "thinking out loud" — narrating
# its deliberation instead of answering. If a reply is saturated with these,
# the model leaked its reasoning into the answer and we must redo the call.
_THINKING_MARKERS = [
    "analyze the user", "scan the manual", "let's try", "let me try",
    "initial thought", "second thought", "drafting the response",
    "final decision", "synthesize the answer", "let's re-read",
    "let me reconsider", "rule 1:", "rule 2:", "i'll go with",
    "let's go with", "final proposed answer", "let me re-evaluate",
]

# A hard non-thinking instruction prepended on a retry. Sarvam's hybrid model
# honours an explicit /no_think style directive in the message itself.
_NO_THINK = (
    "/no_think\n"
    "Answer immediately and directly. Do NOT think out loud, do NOT plan, "
    "do NOT write any analysis, drafts, or numbered reasoning. Output ONLY "
    "the final short answer for the user.\n\n"
)


def _clean(text):
    """Strip any stray <think>...</think> block and surrounding whitespace."""
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def _looks_like_thinking(text: str) -> bool:
    """True if the reply is the model narrating its reasoning, not answering.

    We count how many distinct 'thinking' markers appear. A normal answer has
    none. A leaked-reasoning reply has many, so 2+ is a reliable signal.
    """
    low = text.lower()
    hits = sum(1 for m in _THINKING_MARKERS if m in low)
    return hits >= 2


def _looks_repetitive(text: str) -> bool:
    """True if the answer is a degenerate repetition loop."""
    words = text.split()
    if len(words) <= 60:
        return False
    spans = [" ".join(words[i:i + 6]) for i in range(len(words) - 6)]
    return bool(spans) and (len(set(spans)) / len(spans)) < 0.5


def _post(messages, temperature, max_tokens, retries):
    """Single Sarvam request with network/rate-limit retries. Returns the
    cleaned `content` string, or raises SarvamError."""
    headers = {
        "Authorization": f"Bearer {SARVAM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": SARVAM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # sarvam-30b is a hybrid reasoning model on vLLM; this disables most
        # of the thinking. The prompt-level /no_think directive handles the
        # rest.
        "chat_template_kwargs": {"enable_thinking": False},
    }

    last_err = ""
    for attempt in range(retries):
        try:
            resp = requests.post(SARVAM_API_URL, headers=headers,
                                 json=payload, timeout=120)
        except requests.RequestException as exc:
            last_err = f"network error: {exc}"
            time.sleep(2 ** attempt)
            continue

        if resp.status_code == 200:
            data = resp.json()
            choice = data["choices"][0]
            message = choice.get("message", {}) or {}
            text = _clean(message.get("content")) \
                or _clean(message.get("reasoning_content"))
            if text:
                return text
            raise SarvamError(
                "Sarvam returned an empty answer (finish_reason="
                f"{choice.get('finish_reason')})."
            )

        if resp.status_code in (429, 500, 502, 503, 504):
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            time.sleep(2 ** attempt)
            continue

        raise SarvamError(
            f"Sarvam API error {resp.status_code}: {resp.text[:300]}"
        )

    raise SarvamError(
        f"Sarvam API failed after {retries} attempts. Last error: {last_err}"
    )


def sarvam_chat(messages, temperature=0.0, max_tokens=4000, retries=4):
    """Send a chat-completion request to Sarvam and return the reply text.

    If the model leaks its internal reasoning into the answer, this makes ONE
    corrective retry with a hard no-think instruction. A degenerate repetition
    loop fails cleanly.

    Raises:
        SarvamError: on a missing key, a non-retryable error, or repeated
        failure.
    """
    if not SARVAM_API_KEY:
        raise SarvamError(
            "SARVAM_API_KEY is not set. Create a .env file in the project "
            "root containing:  SARVAM_API_KEY=sk_your_key_here"
        )

    text = _post(messages, temperature, max_tokens, retries)

    # If the model narrated its reasoning instead of answering, retry ONCE
    # with a hard no-think directive injected into the first message.
    if _looks_like_thinking(text):
        retry_messages = [dict(m) for m in messages]
        for m in retry_messages:
            if m.get("role") in ("system", "user"):
                m["content"] = _NO_THINK + m["content"]
                break
        text = _post(retry_messages, temperature, max_tokens, retries)

    if _looks_repetitive(text):
        raise SarvamError(
            "Sarvam returned a repetitive answer. Please rephrase the "
            "question and try again."
        )
    return text
