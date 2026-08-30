import json

with open('MRBench_V2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(len(data))                   # Should be ~200 dialogues
print(data[0].keys())              # Top-level keys
print(list(data[0]['anno_llm_responses'].keys()))  # Tutor models
