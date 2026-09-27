import re
import sys
import os
import numpy as np
import spacy

sys.path.append(os.path.join(os.path.dirname(__file__)))
from stylometric import extract_stylometric_features, get_words, get_sentences

_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return _nlp

DISCOURSE_MARKERS = {
    "however", "furthermore", "moreover", "therefore", "consequently",
    "nevertheless", "nonetheless", "meanwhile", "additionally", "thus",
    "hence", "accordingly", "conversely", "similarly", "in conclusion",
    "in summary", "on the other hand", "as a result", "for instance",
    "for example", "in addition", "in contrast"
}

POS_TAGS_OF_INTEREST = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "CONJ", "PROPN"]

def discourse_marker_rate(text):
    lower = text.lower()
    words = get_words(text)
    n_words = len(words) if words else 1
    count = 0
    for marker in DISCOURSE_MARKERS:
        count += lower.count(marker)
    return {"discourse_marker_rate": count / n_words}

def pos_distribution(text, nlp):
    doc = nlp(text[:1500])  # cap length for speed, avoid pathological slow cases
    total = len(doc) if len(doc) > 0 else 1
    counts = {tag: 0 for tag in POS_TAGS_OF_INTEREST}
    for token in doc:
        if token.pos_ in counts:
            counts[token.pos_] += 1
    return {f"pos_{tag.lower()}_ratio": counts[tag] / total for tag in POS_TAGS_OF_INTEREST}

def extract_attribution_features(text, nlp=None):
    if nlp is None:
        nlp = get_nlp()
    features = extract_stylometric_features(text)
    features.update(discourse_marker_rate(text))
    features.update(pos_distribution(text, nlp))
    return features

