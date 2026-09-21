"""
ToyBot backend — Ollama edition.

Runs the same /chat API your chat widget already talks to, but powered by a
model running locally via Ollama instead of the Gemini cloud API. No API key,
no rate limits, works fully offline.

Setup:
    1. Install Ollama: https://ollama.com
    2. Pull a model (pick one that fits your hardware):
           ollama pull llama3.1:8b      # good default, needs ~6GB RAM/VRAM
           ollama pull qwen2.5:7b       # similar size, strong instruction following
           ollama pull phi4             # smaller, lighter on weaker machines
    3. Make sure Ollama is running (it usually starts automatically after
       install; otherwise run `ollama serve` in a terminal).
    4. pip install flask flask-cors requests
    5. python toybot_backend.py

The widget's TOYBOT_ENDPOINT in toy-store.html should point at:
    http://localhost:5000/chat
(or wherever you end up running this).

Note: for a live demo, run this file and Ollama on the SAME machine you'll
be demoing from, and open toy-store.html on that machine too — that's what
gets you the "zero internet required" setup.
"""

import logging
import os
import uuid

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# In production, restrict this to your storefront's actual domain instead of "*", e.g.:
# CORS(app, resources={r"/*": {"origins": "https://your-store-domain.com"}})
CORS(app)

# ---- Config (override via environment variables if you want to change these
# without editing the file) ----
OLLAMA_CHAT_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_TAGS_URL = os.environ.get("OLLAMA_TAGS_URL", "http://localhost:11434/api/tags")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("OLLAMA_TIMEOUT", "60"))

# How many past turns (user+assistant messages) to keep sending as context.
# Keeps requests fast and bounded instead of growing forever in a long chat.
MAX_HISTORY_MESSAGES = 20

TOYBOT_SYSTEM_INSTRUCTION = """
You are ToyBot, the friendly, energetic, and helpful virtual assistant for ToyStore, an online store specializing in toys, games, and kid-friendly services.

### Tone & Style:
- Enthusiastic, warm, and welcoming—like a cheerful employee at a magical toy shop.
- Keep responses concise, clear, and easy to read (use bullet points for lists of toys).
- Always maintain patience and empathy, especially if a customer is having issues.

### Capabilities:
- Recommend toys based on age groups, interests, or gift occasions.
- Provide information on store shipping policies, gift wrapping, and standard return windows (30 days).
- Help customers find general categories (e.g., board games, educational STEM toys, outdoor playsets).

### Guardrails:
- Do NOT invent fictional prices, discounts, or inventory levels if you don't know them. Direct the user to check the product page.
- Do NOT process real financial transactions or ask for sensitive info like credit card numbers.
- If a user asks about something completely unrelated to ToyStore, politely steer the conversation back to toys and gifts.
"""

# Unlike the Gemini SDK's chat objects, Ollama's /api/chat is stateless — you
# send the full message list every time. So each session here is just a list
# of {"role", "content"} dicts, starting with the system prompt.
sessions = {}


def get_history(session_id):
    if session_id not in sessions:
        sessions[session_id] = [{"role": "system", "content": TOYBOT_SYSTEM_INSTRUCTION}]
    return sessions[session_id]


def trimmed_history(history):
    # Always keep the system message (index 0), then only the most recent
    # turns after it, so long conversations don't slow every request down.
    return [history[0]] + history[1:][-MAX_HISTORY_MESSAGES:]


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())

    if not message:
        return jsonify({"error": "message is required"}), 400

    history = get_history(session_id)
    history.append({"role": "user", "content": message})

    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": trimmed_history(history),
                "stream": False,
                "options": {"temperature": 0.7},
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        reply_text = (payload.get("message") or {}).get("content", "").strip()

        if not reply_text:
            app.logger.warning("Ollama returned no content. Raw response: %r", payload)
            reply_text = "Sorry, I couldn't come up with an answer to that one — could you rephrase?"

        history.append({"role": "assistant", "content": reply_text})
        sessions[session_id] = history

        return jsonify({"reply": reply_text, "session_id": session_id})

    except requests.exceptions.ConnectionError:
        app.logger.exception("Could not reach Ollama")
        return jsonify({
            "error": (
                "Can't reach Ollama at " + OLLAMA_CHAT_URL + ". Is it running? "
                "Open the Ollama app or run `ollama serve`, and make sure the model "
                "is pulled: `ollama pull " + OLLAMA_MODEL + "`."
            )
        }), 503
    except requests.exceptions.Timeout:
        app.logger.exception("Ollama request timed out")
        return jsonify({"error": "ToyBot took too long to respond. Try again."}), 504
    except requests.exceptions.HTTPError as e:
        app.logger.exception("Ollama returned an HTTP error")
        return jsonify({"error": "Ollama error: " + str(e)}), 502
    except Exception as e:
        app.logger.exception("ToyBot /chat failed")
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    """Optional: let the frontend clear a session's memory (e.g. a 'New chat' button)."""
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id")
    if session_id and session_id in sessions:
        del sessions[session_id]
    return jsonify({"status": "reset"})


@app.route("/health", methods=["GET"])
def health():
    ollama_reachable = False
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=3)
        ollama_reachable = r.ok
    except Exception:
        ollama_reachable = False

    return jsonify({
        "status": "ok",
        "model": OLLAMA_MODEL,
        "ollama_reachable": ollama_reachable,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
