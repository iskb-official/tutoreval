# check_gpt4.py
import pandas as pd
df = pd.read_csv("dialogue_analysis.csv")
gpt4_rows = df[df['Model_Name'] == 'GPT4']
for i, row in gpt4_rows.iterrows():
    print(f"GPT4 #{i}: {row['pred_label']} P={row['p_good']:.3f}")
    print(f"  Response: {row['response'][:100]}...")
    print()
