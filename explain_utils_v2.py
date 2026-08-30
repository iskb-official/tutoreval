# explain_utils_v2.py

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# ---------- 1. Load v2 model, vectorizer, encoder, and feature list ----------

# TF-IDF + XGBoost model trained with ped_binary_v2 labels
clf = joblib.load("xgb_ped_binary_v2_tfidf.joblib")
tfidf: TfidfVectorizer = joblib.load("tfidf_ped_binary_v2.joblib")
le = joblib.load("label_encoder_ped_binary_v2.joblib")

# Top 50 global features for explanation
feat_df = pd.read_csv("top50_tfidf_xgb_features.csv")  # feature, importance
TOP_FEATURES = feat_df["feature"].tolist()
FEATURE_IMPORTANCE = dict(zip(feat_df["feature"], feat_df["importance"]))


# ---------- 2. Core prediction + explanation logic ----------

def _predict_label_and_prob(text: str) -> Dict[str, Any]:
    """Predict Good/Poor and P(Good) using the v2 TF-IDF + XGB model."""
    X_vec = tfidf.transform([text])
    probs_full = clf.predict_proba(X_vec)  # shape: [1, n_classes]

    # Find index of "Good" in the label encoder
    good_idx = int(np.where(le.classes_ == "Good")[0][0])
    p_good = float(probs_full[0, good_idx])

    label = "Good" if p_good >= 0.5 else "Poor"
    return {"label": label, "p_good": p_good}


def _match_top_features(text: str) -> List[str]:
    """Return all high-importance n-grams present in the text (case-insensitive)."""
    text_low = text.lower()
    present = []
    for feat in TOP_FEATURES:
        if feat.lower() in text_low:
            present.append(feat)
    # Sort by importance, descending
    present = sorted(present, key=lambda f: FEATURE_IMPORTANCE[f], reverse=True)
    return present


def _build_explanation(label: str, matched_feats: List[str]) -> str:
    """Create a short natural-language explanation based on label + feature matches."""
    if not matched_feats:
        if label == "Good":
            return (
                "Predicted as Good mainly from overall linguistic patterns consistent "
                "with MRBench guidance and actionability, although no top global "
                "scaffolding phrases were detected."
            )
        else:
            return (
                "Predicted as Poor and no strong scaffolding or encouragement phrases "
                "were detected. The response may be answer-only, vague, or weakly "
                "aligned with the student's prior turn."
            )

    top_show = matched_feats[:3]
    feats_str = ", ".join(f"“{f}”" for f in top_show)

    if label == "Good":
        return (
            f"Predicted as Good because it includes pedagogically helpful phrases "
            f"such as {feats_str}, which typically signal guidance, explanation, or "
            f"encouraging tone in MRBench annotations."
        )
    else:
        return (
            f"Predicted as Poor even though it contains phrases like {feats_str}. "
            f"These appear insufficient to provide clear next steps or rich guidance "
            f"for the student."
        )


def explain_response(text: str) -> Dict[str, Any]:
    """
    Full pipeline (v2):
      - Predict Good/Poor and P(Good) using TF-IDF + XGBoost v2
      - Find which high-importance n-grams appear
      - Generate a short explanation string
    """
    pred = _predict_label_and_prob(text)
    matched = _match_top_features(text)
    explanation = _build_explanation(pred["label"], matched)

    return {
        "text": text,
        "label": pred["label"],
        "p_good": pred["p_good"],
        "matched_features": matched,
        "explanation": explanation,
    }


# ---------- 3. Quick manual test ----------

if __name__ == "__main__":
    examples = [
        "Good try! Let's go step by step. First, check whether 12 and 18 share a common factor. "
        "If we divide both by 6, we get 2/3.",
        "The answer is 42.",
    ]

    for t in examples:
        out = explain_response(t)
        print("\n==============================")
        print("Text:", out["text"])
        print(f"Label: {out['label']}  (P(Good) = {out['p_good']:.3f})")
        print("Matched features:", out["matched_features"][:5])
        print("Explanation:", out["explanation"])
