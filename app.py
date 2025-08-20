import os
import re
import json
import pickle
import logging
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "..", "tokenizer.pkl")

# ---- Load model/tokenizer (if available) ----
model = None
tokenizer = None
MAX_LEN = 100  # adjust if you know the training value
START_TOKEN = "sostok"  # common in many seq2seq tutorials
END_TOKEN = "eostok"

try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        app.logger.info("✅ Loaded model.h5")
        app.logger.info("Model expects %d input(s).", len(model.inputs))
    else:
        app.logger.warning("⚠ model.h5 not found at %s", MODEL_PATH)
except Exception as e:
    app.logger.exception("❌ Failed to load model: %s", e)

try:
    if os.path.exists(TOKENIZER_PATH):
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
        app.logger.info("✅ Loaded tokenizer.pkl (vocab size ~ %d)", len(getattr(tokenizer, "word_index", {})))
    else:
        app.logger.warning("⚠ tokenizer.pkl not found at %s", TOKENIZER_PATH)
except Exception as e:
    app.logger.exception("❌ Failed to load tokenizer: %s", e)

# ---- Helpers ----
def simple_extractive_summary(text: str, max_sentences: int = 3) -> str:
    """
    Fallback: frequency-based extractive summary.
    No external downloads; works offline.
    """
    # Split into sentences (simple regex)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]

    if len(sentences) <= max_sentences:
        return text.strip()

    # Word frequencies (ignore tiny stopword list)
    stop = {
        'the','a','an','is','are','am','and','or','but','if','on','in','to','for','of','with','as',
        'by','at','from','this','that','these','those','it','its','be','been','was','were','will',
        'would','can','could','should','I','you','he','she','they','we','my','your','their',
    }
    words = re.findall(r"\w+", text.lower())
    freqs = {}
    for w in words:
        if w.isnumeric() or w in stop:
            continue
        freqs[w] = freqs.get(w, 0) + 1

    # Score sentences
    scores = []
    for i, s in enumerate(sentences):
        sw = re.findall(r"\w+", s.lower())
        score = sum(freqs.get(w, 0) for w in sw)
        scores.append((i, score))

    # Pick top N by score, then sort back to original order
    top_idx = sorted(sorted(scores, key=lambda x: x[1], reverse=True)[:max_sentences], key=lambda x: x[0])
    chosen = [sentences[i] for i, _ in top_idx]
    return " ".join(chosen)

def try_model_summary(text: str) -> str:
    """
    Attempt to summarize using the loaded model & tokenizer.
    Supports single-input models. If the model needs multiple inputs (typical seq2seq),
    we will fail and the caller should fall back to extractive.
    """
    if model is None or tokenizer is None:
        raise RuntimeError("Model or tokenizer not loaded.")

    # Tokenize & pad
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

    # If the model expects 1 input, call directly.
    if len(model.inputs) == 1:
        pred = model.predict(padded)
        # If output is (batch, time, vocab) -> argmax per time-step
        if pred.ndim == 3:
            token_ids = np.argmax(pred, axis=-1)[0]
        # If output is (batch, vocab) classification-style -> single argmax
        elif pred.ndim == 2:
            token_ids = np.argmax(pred, axis=-1)  # shape (batch,)
            token_ids = [int(token_ids[0])]
        else:
            raise RuntimeError(f"Unexpected prediction shape: {pred.shape}")

        # Decode to words
        idx2word = getattr(tokenizer, "index_word", {})
        words = []
        for idx in token_ids:
            if idx == 0:
                continue
            w = idx2word.get(int(idx))
            if not w:
                continue
            if w == END_TOKEN:
                break
            if w == START_TOKEN:
                continue
            words.append(w)
        summary = " ".join(words).strip()
        return summary if summary else "(empty summary)"

    # If we get here, the model likely expects multiple inputs (encoder/decoder).
    raise RuntimeError("Model expects multiple inputs; inference graph not available.")

# ---- Routes ----
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "has_model": model is not None,
        "has_tokenizer": tokenizer is not None,
        "model_inputs": None if model is None else len(model.inputs),
    })

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.get_json(force=True, silent=False)
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "No text provided."}), 400

        # First try the ML model
        try:
            summary = try_model_summary(text)
            # If summary is too short, fall back to extractive to be helpful
            if len(summary.split()) < 5:
                app.logger.info("Model summary looked too short; using fallback.")
                summary = simple_extractive_summary(text, max_sentences=3)
            return jsonify({"summary": summary})

        except Exception as ml_err:
            app.logger.warning("Model summarize failed: %r. Falling back.", ml_err)
            # Fallback: extractive
            summary = simple_extractive_summary(text, max_sentences=3)
            return jsonify({
                "summary": summary,
                "note": "Served by extractive fallback because the model inference failed."
            })

    except Exception as e:
        app.logger.exception("Unhandled error in /summarize: %s", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Helpful logs at startup
    app.logger.info("Starting Flask…")
    app.logger.info("MODEL_PATH: %s", MODEL_PATH)
    app.logger.info("TOKENIZER_PATH: %s", TOKENIZER_PATH)
    app.run(debug=True)
