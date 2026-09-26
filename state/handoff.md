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
Status: not started
Result: [accuracy/F1 from results/metrics/binary.json]
Decision: [target met? floor met? bounced back?]
Notes: 

## Step 6 — Attribution features
Status: not started
Outputs: [top features from ablation.json]
Notes: 

## Step 7 — Attribution model
Status: not started
Model classes chosen: [e.g. GPT / Claude / LLaMA]
Result: [accuracy/macro F1 from attribution.json]
Decision: [target met? floor met? bounced back?]
Notes: 

## Step 8 — Calibration and robustness
Status: not started
Result: [calibration + robustness numbers]
Notes: 

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




