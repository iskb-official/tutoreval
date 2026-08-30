# local_llm_baseline.py
import pandas as pd
import ollama
import time
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 1. Load the test set (or a sample of it)
df = pd.read_csv("MRBench_V2_flat.csv").dropna(subset=['response']).sample(100, random_state=42)

# Determine ground truth (1 for Good, 0 for Poor)
def get_label(row):
    g = str(row.get('Providing_Guidance', '')).strip()
    a = str(row.get('Actionability', '')).strip()
    c = str(row.get('Coherence', '')).strip()
    return 1 if g in ['Yes', 'To some extent'] and a in ['Yes', 'To some extent'] and c == 'Yes' else 0

y_true = df.apply(get_label, axis=1).tolist()
texts = df['response'].tolist()

y_pred_llm = []
start_time = time.time()

# 2. Run Local Zero-Shot Inference
print("Starting local LLM inference...")
for i, text in enumerate(texts):
    prompt = f"""You are an expert AI tutor evaluator. Read the following tutor response. 
Does this response simultaneously provide clear guidance, offer an actionable next step, and maintain coherence?
Respond with EXACTLY the word "Good" if it meets all three criteria, or "Poor" if it fails any.

Tutor Response: "{text}"
Decision:"""

    try:
        response = ollama.chat(model='llama3.2:3b', messages=[{'role': 'user', 'content': prompt}])
        reply = response['message']['content'].strip().lower()
        y_pred_llm.append(1 if "good" in reply else 0)
    except Exception as e:
        print(f"Error on row {i}: {e}")
        y_pred_llm.append(0)

end_time = time.time()

# 3. Output Benchmark Metrics
print("\n=== LOCAL LLM BASELINE ===")
print(f"Accuracy: {accuracy_score(y_true, y_pred_llm):.3f}")
print(f"F1 Score: {f1_score(y_true, y_pred_llm):.3f}")
print(f"Average Inference Time per Response: {(end_time - start_time) / len(texts):.3f} seconds")