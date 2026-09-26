import nltk
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

N = 2  # bigram model

def train_class_lm(texts, n=N):
    tokenized = [word_tokenize(t.lower()) for t in texts]
    train_data, vocab = padded_everygram_pipeline(n, tokenized)
    lm = MLE(n)
    lm.fit(train_data, vocab)
    return lm

def score_text_against_lms(text, lms, n=N):
    tokens = word_tokenize(text.lower())
    scores = {}
    for class_name, lm in lms.items():
        try:
            test_data = list(nltk.lm.preprocessing.padded_everygrams(n, tokens))
            perplexity = lm.perplexity(test_data)
            if perplexity == float("inf") or perplexity != perplexity:  # inf or nan
                perplexity = 1e6
        except Exception:
            perplexity = 1e6
        scores[f"perplexity_{class_name}"] = perplexity
    return scores
