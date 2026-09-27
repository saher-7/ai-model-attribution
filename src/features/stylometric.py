import re
import numpy as np

HEDGE_WORDS = {
    "maybe", "perhaps", "possibly", "might", "could", "seems", "appears",
    "somewhat", "generally", "typically", "often", "usually", "likely",
    "suggests", "probably", "arguably", "presumably", "tends", "roughly"
}

def get_sentences(text):
    return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

def get_words(text):
    return re.findall(r"\b\w+\b", text.lower())

def sentence_length_stats(text):
    sentences = get_sentences(text)
    if not sentences:
        return {"sent_len_mean": 0.0, "sent_len_std": 0.0}
    lengths = [len(get_words(s)) for s in sentences]
    return {
        "sent_len_mean": float(np.mean(lengths)),
        "sent_len_std": float(np.std(lengths)) if len(lengths) > 1 else 0.0
    }

def punctuation_ratios(text):
    n = len(text) if len(text) > 0 else 1
    return {
        "comma_ratio": text.count(",") / n,
        "period_ratio": text.count(".") / n,
        "semicolon_ratio": text.count(";") / n,
        "exclaim_ratio": text.count("!") / n,
        "question_ratio": text.count("?") / n,
    }

def hedge_word_count(text):
    words = get_words(text)
    if not words:
        return {"hedge_rate": 0.0}
    count = sum(1 for w in words if w in HEDGE_WORDS)
    return {"hedge_rate": count / len(words)}

def type_token_ratio(text):
    words = get_words(text)
    if not words:
        return {"type_token_ratio": 0.0}
    return {"type_token_ratio": len(set(words)) / len(words)}

def extract_stylometric_features(text):
    features = {}
    features.update(sentence_length_stats(text))
    features.update(punctuation_ratios(text))
    features.update(hedge_word_count(text))
    features.update(type_token_ratio(text))
    return features
