# make_binary_labels_v2.py

import pandas as pd

df = pd.read_csv("MRBench_V2_flat.csv")

cond_good = (
    (df["Providing_Guidance"].isin(["Yes", "To some extent"])) &
    (df["Actionability"].isin(["Yes", "To some extent"])) &
    (df["Coherence"] == "Yes")
)

df["ped_binary_v2"] = cond_good.map({True: "Good", False: "Poor"})
print(df["ped_binary_v2"].value_counts())

df.to_csv("MRBench_V2_binary_v2.csv", index=False)
print("Saved MRBench_V2_binary_v2.csv")
