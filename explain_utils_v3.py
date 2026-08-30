# explain_utils_v3.py (ONLY MODEL LOADS CHANGED)
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# ---------- V3 MODEL LOADS ----------
clf = joblib.load("xgb_ped_binary_v3_calibrated.joblib")  # CHANGED
tfidf = joblib.load("tfidf_ped_binary_v3.joblib")         # CHANGED
le = joblib.load("label_encoder_ped_binary_v3.joblib")    # CHANGED

# Same top features
feat_df = pd.read_csv("top50_tfidf_xgb_features.csv")
TOP_FEATURES = feat_df["feature"].tolist()
FEATURE_IMPORTANCE = dict(zip(feat_df["feature"], feat_df["importance"]))

# ---------- SAME FUNCTIONS AS V2 ----------
def _predict_label_and_prob(text: str) -> Dict[str, Any]:
    X_vec = tfidf.transform([text])
    probs_full = clf.predict_proba(X_vec)
    good_idx = int(np.where(le.classes_ == "Good")[0][0])
    p_good = float(probs_full[0, good_idx])
    label = "Good" if p_good >= 0.5 else "Poor"
    return {"label": label, "p_good": p_good}

def _match_top_features(text: str) -> List[str]:
    text_low = text.lower()
    present = [feat for feat in TOP_FEATURES if feat.lower() in text_low]
    return sorted(present, key=lambda f: FEATURE_IMPORTANCE[f], reverse=True)

def _build_explanation(label: str, matched_feats: List[str]) -> str:
    if not matched_feats:
        if label == "Good":
            return "Predicted as Good from overall MRBench-aligned patterns (no top features detected)."
        return "Predicted as Poor - lacks strong scaffolding/guidance signals."
    
    top_show = matched_feats[:3]
    feats_str = ", ".join(f"“{f}”" for f in top_show)
    
    if label == "Good":
        return f"Predicted as Good: contains helpful phrases {feats_str} (MRBench guidance signals)."
    return f"Predicted as Poor despite {feats_str} - insufficient for clear next steps."

def explain_response(text: str) -> Dict[str, Any]:
    pred = _predict_label_and_prob(text)
    matched = _match_top_features(text)
    explanation = _build_explanation(pred["label"], matched)
    return {
        "text": text, "label": pred["label"], "p_good": pred["p_good"],
        "matched_features": matched, "explanation": explanation,
    }

if __name__ == "__main__":
    examples = [
        "Good try! Let's go step by step. First, check whether 12 and 18 share a common factor.",
        "Nice effort. Notice that 12 and 18 are both divisible by 6.",
    ]
    for t in examples:
        out = explain_response(t)
        print(f"\nLabel: {out['label']} (P={out['p_good']:.3f})")
        print("Features:", out["matched_features"][:3])
