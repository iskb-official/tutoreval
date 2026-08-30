# analyze_dialogues.py (V4 HYBRID FIXED)
import json
import csv
from pathlib import Path

from explain_utils_v4 import explain_response  # V4 hybrid ML + rule boost

INPUT_JSON = "sample_dialogue.json"          # change to your file
OUTPUT_CSV = "dialogue_analysis.csv"

def main():
    input_path = Path(INPUT_JSON)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_JSON}")

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        dialogues = json.load(f)

    rows = []

    for d in dialogues:
        conv_id = d.get("conversation_id")
        conv_hist = d.get("conversation_history", "")
        data_name = d.get("Data")
        split = d.get("Split")
        topic = d.get("Topic")
        gt_sol = d.get("Ground_Truth_Solution", "")

        anno = d.get("anno_llm_responses", {})
        for model_name, content in anno.items():
            resp_text = content.get("response", "")
            ann = content.get("annotation", {})

            # Run V4 hybrid Good/Poor classifier + explanation
            expl = explain_response(resp_text)

            row = {
                "conversation_id": conv_id,
                "conversation_history": conv_hist,
                "Data": data_name,
                "Split": split,
                "Topic": topic,
                "Ground_Truth_Solution": gt_sol,
                "Model_Name": model_name,
                "response": resp_text,
                # Original MRBench labels (if present)
                "Mistake_Identification": ann.get("Mistake_Identification"),
                "Mistake_Location": ann.get("Mistake_Location"),
                "Revealing_of_the_Answer": ann.get("Revealing_of_the_Answer"),
                "Providing_Guidance": ann.get("Providing_Guidance"),
                "Actionability": ann.get("Actionability"),
                "Coherence": ann.get("Coherence"),
                "Tutor_Tone": ann.get("Tutor_Tone"),
                "humanlikeness": ann.get("humanlikeness"),
                # V4 hybrid model outputs
                "pred_label": expl["label"],          # Good / Poor
                "p_good": expl["p_good"],
                "matched_features": "; ".join(expl["matched_features"]),
                "explanation": expl["explanation"],
            }
            rows.append(row)

    # Write CSV
    if not rows:
        print("No rows generated – check input JSON.")
        return

    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {OUTPUT_CSV}")
    print("✅ V4 hybrid model complete - MRBench aligned!")

if __name__ == "__main__":
    main()
