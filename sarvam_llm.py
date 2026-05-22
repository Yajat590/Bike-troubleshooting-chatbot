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


def _clean(text):
    """Strip any stray <think>...</think> reasoning block and surrounding
    whitespace, so internal thinking can never reach the user even if a
    block leaks through."""
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def _looks_repetitive(text: str) -> bool:
    """True if the answer is a degenerate repetition loop.

    A healthy answer is almost entirely unique phrasing; a loop (the model
    repeating the same sentence dozens of times) has very few unique spans.
    We slide a 6-word window over the text and measure how many spans are
    unique — below 50% means the answer is broken.
    """
    words = text.split()
    if len(words) <= 60:                       # too short to judge — allow it
        return False
    spans = [" ".join(words[i:i + 6]) for i in range(len(words) - 6)]
    return bool(spans) and (len(set(spans)) / len(spans)) < 0.5


def sarvam_chat(messages, temperature=0.2, max_tokens=4000, retries=4):
    """Send a chat-completion request to Sarvam and return the reply text.

    Args:
        messages: list of {"role": "system"|"user"|"assistant", "content": str}
        temperature: 0-2. Lower = more focused/deterministic.
        max_tokens: ceiling on generated tokens. Kept generous on purpose —
            see the note on the payload below.
        retries: how many times to retry on network / rate-limit / server
            errors. An empty or degenerate answer is deliberately NOT retried.

    Returns:
        The assistant's reply as a string.

    Raises:
        SarvamError: on a missing key, a non-retryable error, a degenerate
        answer, or repeated failure.
    """
    if not SARVAM_API_KEY:
        raise SarvamError(
            "SARVAM_API_KEY is not set. Create a .env file in the project "
            "root containing:  SARVAM_API_KEY=sk_your_key_here"
        )

    # The sk_xxx key works as a Bearer token (this matches Sarvam's official
    # cURL example). If you ever see a 403, swap this header for:
    #     "api-subscription-key": SARVAM_API_KEY
    headers = {
        "Authorization": f"Bearer {SARVAM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": SARVAM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # --- Turning OFF the reasoning model's "thinking" ---
        # sarvam-30b is a HYBRID think/non-think model served on vLLM.
        # Sending reasoning_effort=null over raw HTTP does NOT disable
        # thinking. The genuine switch exposed by vLLM is the chat-template
        # flag `enable_thinking`, passed through here. With it false the
        # model writes the answer straight into `content`.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    # NOTE on max_tokens: it is intentionally generous (4000). With thinking
    # off a real answer is only a few hundred tokens, and you are billed for
    # tokens actually generated — so the high ceiling costs nothing and just
    # guarantees the answer can never be truncated.

    last_err = ""
    for attempt in range(retries):
        try:
            resp = requests.post(SARVAM_API_URL, headers=headers,
                                 json=payload, timeout=120)
        except requests.RequestException as exc:
            last_err = f"network error: {exc}"
            time.sleep(2 ** attempt)          # 1s, 2s, 4s, 8s back-off
            continue

        if resp.status_code == 200:
            data = resp.json()
            choice = data["choices"][0]
            message = choice.get("message", {}) or {}

            # With thinking OFF the answer lands directly in `content`.
            text = _clean(message.get("content"))
            if text:
                # Guard against a degenerate repetition loop (the model
                # repeating one sentence many times). Fail cleanly rather
                # than show the user a wall of repeated text. Retrying would
                # only burn credits, so this raises immediately.
                if _looks_repetitive(text):
                    raise SarvamError(
                        "Sarvam returned a repetitive answer. Please "
                        "rephrase the question and try again."
                    )
                return text

            # Empty `content` is NOT a transient error — retrying it just
            # burns more credits for the same result. Try the reasoning
            # field once as a last resort, then fail with a clear message.
            text = _clean(message.get("reasoning_content"))
            if text and not _looks_repetitive(text):
                return text
            raise SarvamError(
                "Sarvam returned an empty answer (finish_reason="
                f"{choice.get('finish_reason')}). The answer was likely "
                "truncated — raise max_tokens or check that thinking is off."
            )

        # 429 = rate limited, 5xx = transient server error -> wait and retry.
        if resp.status_code in (429, 500, 502, 503, 504):
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            time.sleep(2 ** attempt)
            continue

        # Anything else (e.g. 400 bad request, 403 auth) is not retryable.
        raise SarvamError(
            f"Sarvam API error {resp.status_code}: {resp.text[:300]}"
        )

    raise SarvamError(
        f"Sarvam API failed after {retries} attempts. Last error: {last_err}"
    )
