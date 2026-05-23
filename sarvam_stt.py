"""Sarvam Speech-to-Text wrapper.

Important: the STT endpoint uses a DIFFERENT auth header than chat
completions.  Both use the same SARVAM_API_KEY value, but:

  - Chat completions:   Authorization: Bearer <key>
  - Speech-to-text:     api-subscription-key: <key>

Sarvam's default `saarika:v2.5` model auto-detects English plus all major
Indian languages, so we send `language_code="unknown"` and let the model
figure it out — the resulting transcript text then flows into the same
Hindi/Hinglish/English routing the typed-input path already uses.

API reference: https://docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe
"""
import requests

from config import SARVAM_API_KEY
from sarvam_llm import SarvamError

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
# saaras:v3 is Sarvam's newer STT model. Compared with saarika:v2.5 it
# handles code-mixed Hindi/English ("Hinglish") more reliably and degrades
# more gracefully on less-than-pristine browser audio. It also exposes the
# `mode` parameter; we pin it to "transcribe" so what the user said is
# faithfully turned into text (no translation, no transliteration).
SARVAM_STT_MODEL = "saaras:v3"
SARVAM_STT_MODE = "transcribe"


def sarvam_transcribe(audio_bytes: bytes,
                      filename: str = "recording.wav",
                      mime_type: str = "audio/wav",
                      language_code: str = "unknown",
                      timeout: int = 60) -> str:
    """Transcribe a recorded audio clip to text.

    Args:
        audio_bytes: raw audio file contents (st.audio_input gives WAV bytes).
        filename:    name reported to the API — only used for diagnostics.
        mime_type:   MIME type of the upload (matches st.audio_input default).
        language_code: BCP-47 hint, or "unknown" for auto-detect (default).
        timeout:     network timeout in seconds.

    Returns:
        The transcript string (already stripped).

    Raises:
        SarvamError on a missing key, an empty recording, a network failure,
        a non-200 response, or an empty transcript.
    """
    if not SARVAM_API_KEY:
        raise SarvamError(
            "SARVAM_API_KEY is not set. Cannot use speech-to-text."
        )
    if not audio_bytes:
        raise SarvamError("Empty recording — nothing to transcribe.")

    # STT uses a Sarvam-specific header, NOT the OpenAI-style Bearer token.
    headers = {"api-subscription-key": SARVAM_API_KEY}
    files = {"file": (filename, audio_bytes, mime_type)}
    data = {
        "model": SARVAM_STT_MODEL,
        "mode": SARVAM_STT_MODE,
        "language_code": language_code,
    }

    try:
        resp = requests.post(SARVAM_STT_URL, headers=headers,
                             files=files, data=data, timeout=timeout)
    except requests.RequestException as exc:
        raise SarvamError(f"Speech-to-text network error: {exc}")

    if resp.status_code != 200:
        raise SarvamError(
            f"Sarvam STT error {resp.status_code}: {resp.text[:300]}"
        )

    try:
        body = resp.json()
    except ValueError:
        raise SarvamError("Sarvam STT returned a non-JSON response.")

    transcript = (body.get("transcript") or "").strip()
    if not transcript:
        raise SarvamError(
            "Sarvam STT returned an empty transcript — please try again "
            "and speak clearly into the microphone."
        )
    return transcript
