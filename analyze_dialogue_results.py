# analyze_dialogue_results.py (ROBUST VERSION)
import pandas as pd

df = pd.read_csv("dialogue_analysis.csv")
print("Shape:", df.shape)
print("\nColumn names:", df.columns.tolist())

# Inspect pred_label values
print("\nUnique pred_label values:")
print(df['pred_label'].unique()[:10])  # First 10
print("\nSample pred_label:")
print(df['pred_label'].head())

# Simple crosstab (handles any data type)
print("\n=== MODEL PERFORMANCE ===")
model_pred = pd.crosstab(df['Model_Name'], df['pred_label'], normalize='index') * 100
print(model_pred.round(1))

print("\n=== RAW COUNTS ===")
counts = pd.crosstab(df['Model_Name'], df['pred_label'])
print(counts)

print("\n=== MRBench GUIDANCE ===")
print(df.groupby('Model_Name')['Providing_Guidance'].value_counts())
