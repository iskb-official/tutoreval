import json
import csv

input_file = "MRBench_V2.json"          # or your actual filename
output_file = "MRBench_V2_flat.csv"

# Load JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for d in data:
    base = {
        "conversation_id": d.get("conversation_id"),
        "conversation_history": d.get("conversation_history"),
        "Data": d.get("Data"),
        "Split": d.get("Split"),
        "Topic": d.get("Topic"),
        "Ground_Truth_Solution": d.get("Ground_Truth_Solution"),
    }

    # Iterate over each tutor model for this dialogue
    for model_name, content in d.get("anno_llm_responses", {}).items():
        row = base.copy()
        row["Model_Name"] = model_name
        row["response"] = content.get("response")

        ann = content.get("annotation", {})
        for k, v in ann.items():
            row[k] = v

        rows.append(row)

# Get column names from first row
fieldnames = list(rows[0].keys())

# Write CSV
with open(output_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {output_file}")
