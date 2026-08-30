import pandas as pd

df = pd.read_csv("MRBench_V2_preprocessed.csv")

cond_good = (
    (df["Providing_Guidance_num"] >= 1) &
    (df["Actionability_num"] >= 1) &
    (df["Coherence_num"] == 2)
)
df["ped_binary"] = cond_good.map({True: "Good", False: "Poor"})

print(df["ped_binary"].value_counts())
df.to_csv("MRBench_V2_binary.csv", index=False)
print("Saved MRBench_V2_binary.csv")
