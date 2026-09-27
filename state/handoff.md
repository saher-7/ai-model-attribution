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
Status: done (final, after 2 post-hoc tuning attempts beyond initial fix)
Model classes chosen: GPT-4o, Llama-8B, Mistral-7B, Qwen-2-72B
Result: FINAL MODEL = LightGBM + log-scaled perplexity. Val acc 73.99%25, val macro F1 73.94%25; test acc 61.00%25, test macro F1 59.48%25
Decision: target (55%25) and floor (40%25) both cleared with strong margin. Progression: LogisticRegression+scaling (test acc 53.68%25/F1 50.01%25) -> LightGBM swap (test acc 60.79%25/F1 59.23%25, big jump) -> LightGBM+perplexity (test acc 61.00%25/F1 59.48%25, small further gain). Stopped per spec Section 4 (2-attempt cap) after this.
Notes: Confusion matrix (final model): GPT-4o 92%25 correct, Mistral-7B 69%25 correct, Llama-8B 47%25 correct, Qwen-2-72B 35%25 correct (up from 18%25 with LogisticRegression - still weakest, mostly confused with Mistral-7B - reportable finding, not pursued further). Models/metrics for all 3 attempts saved in results/ for comparison in paper.

## Step 8 — Calibration and robustness
Status: done
Result: Calibration - post-cal test acc 48.84%25, macro F1 44.28%25, mean confidence 47.18%25 (close to actual acc, reasonably honest). Robustness - accuracy drop only 0.2pts under light paraphrasing (55.20%25 -> 55.00%25)
Notes: Robustness test used light word-swap/sentence-reorder paraphrasing only (n=500 sample) - not tested against heavier paraphrasing. Flag this limitation in write-up.

## Step 9 — 6-model stretch (optional)
Status: skipped
Result: 
Notes: 

## Step 10 — Paper draft
Status: Full prose drafted for Problem Statement, Related Work (MGTBench, M4, MULTITuDE, RAID, HC3, Defactify baseline), Method, Results, and Limitations. Only Conclusion left as TODO (depends on Step 11).
Notes: All numbers in Results/Method pulled directly from results/metrics/*.json and state/handoff.md, not retyped from memory. Run log backfilled with dated entries covering Steps 0-8.

## Step 11 — Final write-up
Status: not started
Notes: 

## Currently blocked on / needs a decision
Nothing blocking. Core deliverable complete (binary + attribution both meet target/floor, calibration + robustness done). Currently attempting optional improvement: LightGBM swap for attribution model (attempt 1 of 2 allowed tuning attempts), possibly paired with properly-scaled perplexity feature (attempt 2) if attempt 1 looks promising. Per spec Section 4, stop after 2 attempts regardless of outcome and move to Step 10 (paper draft). Qwen-2-72B vs Mistral-7B confusion may be a genuine hard case worth reporting as a finding rather than continuing to chase.

## Full results summary (as of this point)
BINARY (Step 5): val acc 91.26%25/F1 95.04%25, test acc 88.23%25/F1 93.18%25. Target (90%25) met on val, floor (80%25) cleared on test.
ABLATION (Step 6): best feature set minus_perplexity at 44.7%25 (sample, 7-class, chance=14%25); raw all_features (unscaled perplexity) worse at 35.4%25.
ATTRIBUTION (Step 7): 4-class (GPT-4o, Llama-8B, Mistral-7B, Qwen-2-72B). Val acc 66.05%25/F1 65.94%25, test acc 53.68%25/F1 50.01%25. Target (55%25) met on val, floor (40%25) cleared on test. Confusion matrix: GPT-4o 89%25 correct, Mistral-7B 70%25 correct, Llama-8B 38%25 correct, Qwen-2-72B only 18%25 correct (mostly confused with Mistral-7B). Fix applied: StandardScaler (unscaled version had macro F1 37%25, below floor - this was the 1 tuning attempt used in Step 7).
CALIBRATION (Step 8): post-cal test acc 48.84%25, macro F1 44.28%25, mean confidence 47.18%25 (close to actual acc - reasonably honest).
ROBUSTNESS (Step 8): original acc 55.20%25/F1 51.18%25 vs paraphrased acc 55.00%25/F1 51.01%25 - only 0.2pt drop. Caveat: light word-swap/sentence-reorder only, not aggressive paraphrasing.
DECISION: All Spec Section 3 targets/floors met. Complete presentable result achieved per Section 4. Step 9 (6-model stretch) skipped as optional/not required. Currently trying 1-2 more tuning attempts on attribution model before moving to write-up.












