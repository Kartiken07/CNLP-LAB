"""Flask app serving the news classifier."""
import os
import sys

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import model_info, predict  # noqa: E402

app = Flask(__name__)
MAX_CHARS = 20000


@app.get("/")
def index():
    return render_template("index.html", info=model_info())


@app.get("/api/health")
def health():
    return jsonify(status="ok")


@app.get("/api/model")
def model():
    return jsonify(model_info())


@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify(error="Please provide some article text."), 400
    if len(text) > MAX_CHARS:
        return jsonify(error=f"Text is too long (max {MAX_CHARS} characters)."), 413
    try:
        return jsonify(predict(text))
    except ValueError as e:
        return jsonify(error=str(e)), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
