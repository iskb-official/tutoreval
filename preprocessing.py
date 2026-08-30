import pandas as pd

df = pd.read_csv("MRBench_V2_flat.csv")

map3 = {"No": 0, "To some extent": 1, "Yes": 2}
map_tone = {"Offensive": 0, "Neutral": 1, "Encouraging": 2}

ann_cols = [
    "Mistake_Identification",
    "Mistake_Location",
    "Providing_Guidance",
    "Actionability",
    "humanlikeness",      # <- use lowercase here
    "Coherence",
]

for col in ann_cols:
    df[col + "_num"] = df[col].map(map3)

df["Tutor_Tone_num"] = df["Tutor_Tone"].map(map_tone)

num_cols = [c for c in df.columns if c.endswith("_num")]
df["ped_score"] = df[num_cols].sum(axis=1)

df["ped_level"] = pd.cut(
    df["ped_score"],
    bins=[-1, 6, 11, 16],   # example thresholds; can adjust after inspection
    labels=["Low", "Medium", "High"]
)

print(df["ped_level"].value_counts())
df.to_csv("MRBench_V2_preprocessed.csv", index=False)
print("Saved MRBench_V2_preprocessed.csv")
