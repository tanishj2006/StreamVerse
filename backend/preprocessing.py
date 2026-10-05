"""
StreamVerse — preprocessing.

These functions are copied VERBATIM from the Colab notebook
(Section 7 for Gujarati, Section 26 for English).

Do not "improve" them. Any divergence between training-time and serving-time
preprocessing causes silent accuracy loss that is very hard to debug.

Stopword lists are NOT hardcoded here — they are read from the bundle's config,
so they can never drift from the lists used during training.
"""

import re
import unicodedata

# --- Gujarati (notebook Section 7) ---------------------------------------
GUJARATI_BLOCK = re.compile(r"[^઀-૿\s]")
MULTISPACE = re.compile(r"\s+")

# --- English (notebook Section 26) ---------------------------------------
NON_ALNUM_EN = re.compile(r"[^a-z0-9\s]")

# --- Script detection (notebook Section 28) ------------------------------
GUJARATI_CHARS = re.compile(r"[઀-૿]")
LATIN_CHARS = re.compile(r"[A-Za-z]")


def clean_text(text):
    """Gujarati: NFC-normalise, keep only the Gujarati block, collapse whitespace."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = GUJARATI_BLOCK.sub(" ", text)
    text = MULTISPACE.sub(" ", text)
    return text.strip()


def preprocess(text, stopwords):
    """Gujarati pipeline: clean -> tokenise -> remove stopwords. Returns a token list."""
    return [t for t in clean_text(text).split() if t not in stopwords]


def clean_text_en(text):
    """English: NFKC-normalise, lowercase, keep a-z0-9, collapse whitespace."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    text = NON_ALNUM_EN.sub(" ", text)
    text = MULTISPACE.sub(" ", text)
    return text.strip()


def preprocess_en(text, stopwords):
    """English pipeline: clean -> lowercase -> tokenise -> remove stopwords."""
    return [t for t in clean_text_en(text).split() if t not in stopwords]


def detect_language(text):
    """Routing rule, exactly as documented in notebook Section 28.

    Any Gujarati character routes to the Gujarati model. Otherwise any Latin
    letter routes to English. Anything else is unroutable.
    """
    text = str(text)
    if GUJARATI_CHARS.search(text):
        return "gu"
    if LATIN_CHARS.search(text):
        return "en"
    return "unknown"
