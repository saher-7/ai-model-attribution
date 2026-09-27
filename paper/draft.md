# AI Model Attribution — Technical Report (Draft)

## 1. Problem Statement

[TODO: 2-3 paragraphs. State the two questions this project answers: (1) is a piece of text human- or AI-written, and (2) if AI-written, which model family produced it. Note the scope: English only, 100-1000 word passages, a fixed set of model families. Note explicitly what is NOT claimed: this is not a 90%+-confidence-for-every-model system, not a production tool, not version detection.]

## 2. Related Work

[TODO: Cite and briefly describe MGTBench, M4, MULTITuDE, RAID, and the NYT/Defactify baseline paper (Rajarshi-Roy-research/Defactify_Text_Dataset - the dataset this project uses). Note the Defactify paper's own baseline: 53% binary accuracy, 5.04% attribution accuracy - this is the number our Results section compares against.]

## 3. Method

[TODO after Step 7 finalized - already done. Write this section now. Cover: dataset (Defactify, 4 selected classes), features (stylometric + POS distribution + discourse markers + per-class bigram perplexity, log-scaled), final model (LightGBM, chosen after comparing to logistic regression), and the binary detector (logistic regression on stylometric features).]

## 4. Results

[TODO after Step 8 - already done. Pull numbers directly from results/metrics/*.json files, never retype from memory. Cover: binary detection numbers, attribution numbers (with confusion matrix), calibration curve, robustness test. Compare directly to Defactify's published baseline (53% binary, 5.04% attribution) - both of our numbers beat it substantially.]

## 5. Limitations

[TODO - build this from logs/run_log.md entries, ongoing. Known limitations so far: Qwen-2-72B vs Mistral-7B confusion persists in final model; robustness test only used light paraphrasing, not aggressive; ablation study used a sample not the full dataset; ]

## 6. Conclusion

[TODO after Step 11]
