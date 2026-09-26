from datasets import load_from_disk
import collections

SEED = 42

ds = load_from_disk("data/raw/attribution_dataset")

print("Splits:", ds)

for split_name in ["train", "validation", "test"]:
    split = ds[split_name]
    out_name = "val" if split_name == "validation" else split_name
    print(f"\n--- {split_name} ---")
    print(f"Rows: {len(split)}")

    for col in ["Label_A", "Label_B"]:
        counts = collections.Counter(split[col])
        print(f"{col} counts:")
        for k, v in sorted(counts.items(), key=lambda x: str(x[0])):
            print(f"  {k}: {v}")

    df = split.to_pandas()
    df.to_csv(f"data/splits/{out_name}.csv", index=False)
    print(f"Saved to data/splits/{out_name}.csv")
