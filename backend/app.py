"""
StreamVerse — FastAPI backend.

Serves the intent classifier and the static frontend from one origin, so there
is no CORS configuration to get wrong on demo day.

Run:  cd backend && uvicorn app:app --reload --port 8000
Open: http://127.0.0.1:8000
"""

import itertools
import os
from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from preprocessing import detect_language, preprocess, preprocess_en

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
FRONTEND_DIR = BASE_DIR.parent / "frontend"
BUNDLE_PATH = MODEL_DIR / "streamverse_bundle.pkl"

# Notebook Section 18. Not stored in the bundle, so it is restated here.
FALLBACK_RESPONSE = "માફ કરશો, હું તમારો પ્રશ્ન સમજી શક્યો નહીં. કૃપા કરીને ફરીથી પૂછો."

# Fixed display order so the intent ladder in the UI never re-sorts between turns.
DEFAULT_INTENT_ORDER = [
    "Information Query",
    "Content Search",
    "Music Recommendation",
    "Movie Recommendation",
    "TV Show Recommendation",
    "Podcast Recommendation",
]


# --------------------------------------------------------------------------
# Load the bundle once at import time.
# --------------------------------------------------------------------------
class Artefacts:
    """Holds everything loaded from streamverse_bundle.pkl, or records why it failed."""

    def __init__(self):
        self.ready = False
        self.error = None
        self.models = {}
        self.stopwords = {}
        self.responses = {}
        self.intent_order = DEFAULT_INTENT_ORDER
        self.sklearn_version = None
        self._rotation = {}

    def load(self):
        if not BUNDLE_PATH.exists():
            self.error = (
                "streamverse_bundle.pkl not found. Export it from the Colab notebook "
                "and place it in backend/model/."
            )
            return self

        try:
            bundle = joblib.load(BUNDLE_PATH)
            cfg = bundle["config"]

            self.models = {
                "gu": (bundle["gu"]["model"], bundle["gu"]["vectorizer"]),
                "en": (bundle["en"]["model"], bundle["en"]["vectorizer"]),
            }
            self.stopwords = {
                "gu": set(cfg["gu"]["stopwords"]),
                "en": set(cfg["en"]["stopwords"]),
            }
            self.responses = bundle["responses"]

            labels = bundle.get("intent_labels") or cfg.get("intent_labels")
            if labels:
                # Keep the canonical order, appending anything unexpected rather than dropping it.
                ordered = [l for l in DEFAULT_INTENT_ORDER if l in labels]
                ordered += [l for l in labels if l not in ordered]
                self.intent_order = ordered

            self._rotation = {k: itertools.cycle(range(len(v)))
                              for k, v in self.responses.items()}

            import sklearn
            self.sklearn_version = sklearn.__version__
            self.ready = True
        except Exception as exc:  # corrupt pickle, sklearn version mismatch, missing key
            self.error = "%s: %s" % (type(exc).__name__, exc)
        return self

    def reply_for(self, intent):
        """Deterministic rotation through an intent's templates (notebook Section 18)."""
        if intent is None or intent not in self.responses:
            return FALLBACK_RESPONSE
        options = self.responses[intent]
        return options[next(self._rotation[intent]) % len(options)]


ART = Artefacts().load()

app = FastAPI(title="StreamVerse", docs_url="/api/docs", redoc_url=None)


class Query(BaseModel):
    text: str = Field(default="", max_length=500)


@app.get("/api/health")
def health():
    """The frontend calls this on load to decide whether to show the offline state."""
    return {
        "ready": ART.ready,
        "error": ART.error,
        "intents": ART.intent_order,
        "sklearn_version": ART.sklearn_version,
        "vocabulary": {
            lang: len(vec.vocabulary_) for lang, (_, vec) in ART.models.items()
        } if ART.ready else {},
    }


@app.post("/api/predict")
def predict(q: Query):
    if not ART.ready:
        return JSONResponse(status_code=503, content={"detail": ART.error})

    raw = (q.text or "").strip()
    if not raw:
        return JSONResponse(status_code=400, content={"detail": "Type a question first."})

    language = detect_language(raw)

    if language == "unknown":
        return {
            "raw": raw, "language": language, "intent": None,
            "response": FALLBACK_RESPONSE, "confidence": None,
            "margin": None, "distribution": [], "tokens": [],
            "note": "No Gujarati or Latin letters found, so there is nothing to classify.",
        }

    model, vectorizer = ART.models[language]
    tokenizer = preprocess if language == "gu" else preprocess_en
    tokens = tokenizer(raw, ART.stopwords[language])

    if not tokens:
        return {
            "raw": raw, "language": language, "intent": None,
            "response": FALLBACK_RESPONSE, "confidence": None,
            "margin": None, "distribution": [], "tokens": [],
            "note": "Every word was a stopword, so no features remain.",
        }

    probs = model.predict_proba(vectorizer.transform([" ".join(tokens)]))[0]
    by_intent = dict(zip(model.classes_, (float(p) for p in probs)))

    ranked = sorted(by_intent.values(), reverse=True)
    top = ranked[0]
    margin = round(top - ranked[1], 4) if len(ranked) > 1 else None
    intent = max(by_intent, key=by_intent.get)

    # Fixed order, so the ladder rows stay put and only the bars move.
    distribution = [
        {"intent": name, "probability": round(by_intent.get(name, 0.0), 4)}
        for name in ART.intent_order
    ]

    # How many of this query's tokens the model has actually seen before. A query
    # made entirely of unseen words falls back to the uniform prior.
    known = sum(1 for t in tokens if t in vectorizer.vocabulary_)

    return {
        "raw": raw,
        "language": language,
        "intent": intent,
        "response": ART.reply_for(intent),
        "confidence": round(top, 4),
        "margin": margin,
        "distribution": distribution,
        "tokens": tokens,
        "known_tokens": known,
        "note": None,
    }


# --- Static frontend (mounted last so /api/* wins) -------------------------
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")

    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    print("Bundle:", "loaded" if ART.ready else "NOT LOADED — %s" % ART.error)
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
