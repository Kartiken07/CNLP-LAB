"""Text preprocessing shared by training and inference."""
import re
from functools import lru_cache

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP_WORDS = frozenset(ENGLISH_STOP_WORDS)
_stemmer = PorterStemmer()

_URL = re.compile(r"http\S+|www\S+")
_HTML = re.compile(r"<[^>]+>")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    return _stemmer.stem(word)


def clean_text(text) -> str:
    """Lowercase, strip URLs/HTML/special chars, drop stopwords, stem.

    The raw AG News text contains literal backslashes used as line breaks
    (e.g. "dwindling\\band"); these are turned into spaces so words are not glued.
    Empty or malformed input returns an empty string.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().replace("\\", " ")
    text = _URL.sub(" ", text)
    text = _HTML.sub(" ", text)
    text = _NON_ALPHA.sub("", text)
    text = _SPACES.sub(" ", text).strip()
    return " ".join(_stem(w) for w in text.split() if w not in STOP_WORDS and len(w) > 1)
