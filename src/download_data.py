from datasets import load_dataset

ds = load_dataset("Rajarshi-Roy-research/Defactify_Text_Dataset")
ds.save_to_disk("data/raw/attribution_dataset")
