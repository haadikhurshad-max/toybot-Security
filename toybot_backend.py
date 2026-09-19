"""
ToyBot backend — a small Flask API that the storefront's chat widget talks to.

Why this exists: the widget on the website runs in the customer's browser, so
it can never hold your Gemini API key directly (anyone could open dev tools
and steal it). This server holds the key instead, and the browser only ever
talks to this server.

Setup:
    pip install flask flask-cors google-genai

    export GEMINI_API_KEY="AQ.Ab8RN6IhMOCHiJM4mc_8GJ-AlqSXrxnQ76LbunaQplqI3Nj_PA"   # rotate the old one first
    python toybot_backend.py

Then point TOYBOT_ENDPOINT in toy-store.html at wherever you deploy this,
e.g. "https://your-app.onrender.com/chat".

Deploying: any host that runs a long-lived Python process works (Render,
Railway, Fly.io, a small VPS, Cloud Run, etc.). Streamlit Community Cloud
won't work for this specific file since it isn't a Streamlit app — it's a
plain HTTP API.
"""

import os
import uuid

from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types

app = Flask(__name__)

# In production, restrict this to your storefront's actual domain instead of "*", e.g.:
# CORS(app, resources={r"/chat": {"origins": "https://your-store-domain.com"}})
CORS(app)

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

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Export it as an environment variable before starting the server "
        "(never hardcode it in source)."
    )

client = genai.Client(api_key=API_KEY)

# Very simple in-memory session store: {session_id: chat_object}.
# This keeps each visitor's conversation on-topic without resending the
# whole history on every request. Two things to know:
#  - It resets whenever the server restarts.
#  - It only works with a single server process. If you deploy with multiple
#    workers/instances behind a load balancer, swap this for a shared store
#    (Redis, a database) keyed by session_id.
sessions = {}


def get_chat(session_id):
    if session_id not in sessions:
        sessions[session_id] = client.chats.create(
            model="gemini-3.6-flash",
            config=types.GenerateContentConfig(
                system_instruction=TOYBOT_SYSTEM_INSTRUCTION,
                temperature=0.7,
            ),
        )
    return sessions[session_id]


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())

    if not message:
        return jsonify({"error": "message is required"}), 400

    try:
        chat_session = get_chat(session_id)
        response = chat_session.send_message(message)
        return jsonify({"reply": response.text, "session_id": session_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
