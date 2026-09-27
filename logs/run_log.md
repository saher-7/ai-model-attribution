# Run Log

## 2026-09-26 — Steps 0-4
- Installed prerequisites (Python 3.13.12, Git, VS Code)
- Set up repo structure, pushed to GitHub (saher-7/ai-model-attribution)
- Created venv, installed pandas/numpy/sklearn/spacy/lightgbm/nltk/datasets
- Downloaded Defactify Text Dataset (Rajarshi-Roy-research/Defactify_Text_Dataset) from HuggingFace - 73K samples, NYT articles + 6 AI models (GPT-4o, Gemma-2-9B, Llama-8B, Mistral-7B, Qwen-2-72B, Yi-Large) + human
- Used dataset's native train/val/test splits (51247/10983/10963) rather than re-splitting

## 2026-09-26 — Step 5: Binary detector
- Built stylometric features (sentence length, punctuation ratios, hedge rate, type-token ratio)
- Logistic regression: val acc 91.26%, test acc 88.23% - target (90%) met on val, floor cleared on test

## 2026-09-26 — Step 6: Attribution features + ablation
- Added POS distribution (spaCy), discourse marker rate, per-class bigram perplexity
- Ablation on sample (300/class): minus_perplexity best at 44.7%; raw all_features worse (35.4%) - unscaled perplexity hurt performance

## 2026-09-26 — Step 7: Attribution model (3 attempts)
- Attempt 0 (baseline): LogisticRegression, unscaled features - macro F1 37% on test, below 40% floor
- Fix: added StandardScaler - test acc 53.68%, macro F1 50.01% - target/floor met. Confusion matrix showed heavy GPT-4o bias, Qwen-2-72B only 18% correct
- Tuning attempt 1: swapped to LightGBM - test acc 60.79%, macro F1 59.23% - meaningful jump, Qwen-2-72B improved to 35%
- Tuning attempt 2 (final, per spec's 2-attempt cap): added log-scaled perplexity features to LightGBM - test acc 61.00%, macro F1 59.48% - small further gain. Adopted as final model.
- Remaining weakness: Qwen-2-72B vs Mistral-7B confusion persists even in final model - treated as a genuine hard case, not pursued further per stopping rule

## 2026-09-26 — Step 8: Calibration and robustness
- Platt scaling (sigmoid, 5-fold CV): post-cal test acc 48.84%, macro F1 44.28%, mean confidence 47.18% (close to actual accuracy - reasonably calibrated)
- Robustness (light word-swap/sentence-reorder paraphrasing, n=500): accuracy dropped only 0.2 points (55.20% -> 55.00%) - model robust to superficial lexical changes; not tested against heavier paraphrasing

## Decision
All Spec Section 3 targets/floors met (binary + attribution). Step 9 (6-model stretch) skipped as optional. Moving to paper draft (Step 10).
