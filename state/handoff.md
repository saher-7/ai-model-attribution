# Handoff State

## Step 0 — Prerequisites
Status: done
Done by: saher
Notes: Python 3.13.12, Git 2.51.0, VS Code 1.138.0 confirmed

## Step 1 — Git repo and structure
Status: done
Outputs: [folders/files created]
Notes: 

## Step 2 — Virtual environment
Status: done
Outputs: requirements.txt committed - y
Notes: 

## Step 3 — Data collection
Status: done
Outputs: Rajarshi-Roy-research/Defactify_Text_Dataset -> data/raw/attribution_dataset
Notes: Already comes with train/validation/test splits (51247/10983/10963); 6 model classes (Gemma-2-9B, Mistral-7B, Qwen-2-72B, LLaMA-8B, Yi-Large, GPT-4o) + human

## Step 4 — Data split
Status: done
Outputs: data/splits/train.csv, val.csv, test.csv (dataset's native splits used, not re-split); SEED = 42
Notes: Label_A = binary human/AI, Label_B = 7-class (6 models + Human_Story), balanced ~7321/class in train

## Step 5 — Binary detector
Status: done
Result: val acc 91.26%25, val F1 95.04%25; test acc 88.23%25, test F1 93.18%25
Decision: target met (val >=90%25), no tuning/bounce-back needed
Notes: Logistic regression on stylometric features (sentence length, punctuation ratios, hedge rate, type-token ratio)

## Step 6 — Attribution features
Status: done
Outputs: minus_perplexity best (44.7%25 on sample, 7-class, chance=14%25); all_features worse (35.4%25) - perplexity needs scaling
Notes: Ablation run on sample (300/class train, ~100/class val), not full data. Features: stylometric + POS dist + discourse markers + per-class bigram perplexity

## Step 7 — Attribution model
Status: done
Model classes chosen: GPT-4o, Llama-8B, Mistral-7B, Qwen-2-72B
Result: val acc 66.05%25, val macro F1 65.94%25; test acc 53.68%25, test macro F1 50.01%25
Decision: target met (val), floor cleared with margin (test). 1 tuning attempt needed (StandardScaler fixed GPT-4o bias; unscaled version had macro F1 37%25, below floor)
Notes: Confusion matrix shows GPT-4o (89%25) and Mistral-7B (70%25) well-separated; Qwen-2-72B (18%25) often confused with Mistral-7B; Llama-8B (38%25) moderate

## Step 8 — Calibration and robustness
Status: done
Result: Calibration - post-cal test acc 48.84%25, macro F1 44.28%25, mean confidence 47.18%25 (close to actual acc, reasonably honest). Robustness - accuracy drop only 0.2pts under light paraphrasing (55.20%25 -> 55.00%25)
Notes: Robustness test used light word-swap/sentence-reorder paraphrasing only (n=500 sample) - not tested against heavier paraphrasing. Flag this limitation in write-up.

## Step 9 — 6-model stretch (optional)
Status: skipped
Result: 
Notes: 

## Step 10 — Paper draft
Status: [sections written so far]
Notes: 

## Step 11 — Final write-up
Status: not started
Notes: 

## Currently blocked on / needs a decision
[Leave empty if nothing. Otherwise: describe exactly what the next agent needs to resolve before continuing.]








