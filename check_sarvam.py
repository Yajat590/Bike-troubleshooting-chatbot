"""One-shot Sarvam API check — run this BEFORE re-testing the app.

It makes a SINGLE short request using the SAME thinking-disabled config the
app uses, and reports how much internal "reasoning" the model produced.

    python check_sarvam.py

How to read the verdict:
    reasoning chars = 0 (or tiny)  -> thinking is OFF. The fix worked.
    reasoning chars = large        -> thinking is STILL ON. Report back
                                      before testing the app — do not spend
                                      credits on the full test yet.
"""
import json

import requests

from config import SARVAM_API_URL, SARVAM_MODEL, SARVAM_API_KEY

if not SARVAM_API_KEY:
    raise SystemExit("SARVAM_API_KEY is not set — check your .env file.")

# Same payload shape the app uses, so this check is meaningful.
payload = {
    "model": SARVAM_MODEL,
    "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
    "temperature": 0.0,
    "max_tokens": 4000,
    "chat_template_kwargs": {"enable_thinking": False},
}

resp = requests.post(
    SARVAM_API_URL,
    headers={"Authorization": f"Bearer {SARVAM_API_KEY}",
             "Content-Type": "application/json"},
    json=payload, timeout=120,
)

print("HTTP status:", resp.status_code)
print("-" * 60)

try:
    data = resp.json()
except ValueError:
    print("Response was not JSON:\n", resp.text[:600])
    raise SystemExit(1)

if resp.status_code != 200:
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print("-" * 60)
    print("RESULT: request failed. If the error mentions "
          "'chat_template_kwargs' or an unknown field, report this back.")
    raise SystemExit(1)

choice = data["choices"][0]
message = choice.get("message", {}) or {}
content = message.get("content")
reasoning = message.get("reasoning_content") or ""
usage = data.get("usage", {}) or {}

print("SUMMARY")
print("  finish_reason     :", choice.get("finish_reason"))
print("  content           :", repr(content))
print("  completion_tokens :", usage.get("completion_tokens"))
print("  reasoning chars   :", len(reasoning))
if reasoning:
    print("  reasoning preview :", repr(reasoning[:160]))
print("-" * 60)

# A one-word reply should need almost no generation. A large reasoning_content
# (or a high completion_tokens count) means thinking is still running.
thinking_off = len(reasoning) < 50 and (usage.get("completion_tokens") or 0) < 50

if thinking_off:
    print("RESULT: thinking is OFF. The fix worked — go ahead and re-test "
          "the app.")
else:
    print("RESULT: thinking is STILL ON (the model produced reasoning for a "
          "one-word answer). The app will still work on the generous token "
          "budget, but report this output back before spending credits on "
          "the full test.")
