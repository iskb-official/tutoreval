import pandas as pd
import numpy as np
import scipy.sparse as sp
from sentence_transformers import SentenceTransformer
import joblib
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

def load_dataset(filepath):
    print("Loading data and Ground Truth labels...")
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['response'])
    
    def determine_label(row):
        guidance = str(row.get('Providing_Guidance', '')).strip()
        actionability = str(row.get('Actionability', '')).strip()
        coherence = str(row.get('Coherence', '')).strip()
        if guidance in ['Yes', 'To some extent'] and actionability in ['Yes', 'To some extent'] and coherence == 'Yes':
            return 1
        return 0

    df['ground_truth'] = df.apply(determine_label, axis=1)
    return df

def apply_pedagogical_rules(texts, base_probs):
    print("Applying Pedagogical Rule Boosting...")
    final_preds = []
    scaffolding_phrases = ["good try", "nice effort", "nice job", "well done", 
                           "step by step", "let us think", "try using", 
                           "remember that", "notice that", "good question", 
                           "good start", "check again"]
    
    for i, text in enumerate(texts):
        text_lower = str(text).lower()
        p_ml = base_probs[i][1] 
        
        c = sum(1 for phrase in scaffolding_phrases if phrase in text_lower)
        boost = min(0.15 * c, 0.5)
        p_final = min(p_ml + boost, 0.95)
        
        final_preds.append(1 if p_final >= 0.5 else 0)
        
    return np.array(final_preds)

def main():
    DATA_PATH = "MRBench_V2_flat.csv"
    df = load_dataset(DATA_PATH)
    texts = df['response'].tolist()
    y_true = df['ground_truth'].values
    
    print("Loading Ultimate Hybrid models from disk...")
    try:
        xgb_model = joblib.load('ultimate_xgb_model.pkl')
        tfidf = joblib.load('ultimate_tfidf.pkl')
    except FileNotFoundError:
        print("Error: Could not find 'ultimate_xgb_model.pkl' or 'ultimate_tfidf.pkl'.")
        return

    print("Extracting TF-IDF features...")
    X_tfidf = tfidf.transform(texts)
    
    print("Extracting Semantic Embeddings (Contextual)...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    X_dense = encoder.encode(texts, show_progress_bar=True)
    
    print("Concatenating Features...")
    X_combined = sp.hstack([X_tfidf, X_dense], format='csr')
    
    print("Running XGBoost Inference...")
    base_probs = xgb_model.predict_proba(X_combined)
    
    y_pred = apply_pedagogical_rules(texts, base_probs)
    
    print("\n==========================================")
    print(" TRUE HUMAN-AI MODEL ALIGNMENT (HYBRID) ")
    print("==========================================")
    kappa_ai = cohen_kappa_score(y_true, y_pred)
    acc_ai = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Full Dataset Size (N): {len(df)}")
    print(f"Human-AI Accuracy / Agreement: {acc_ai * 100:.2f}%")
    print(f"Human-AI Cohen's Kappa (κ): {kappa_ai:.3f}")
    print(f"Confusion Matrix [TN, FP / FN, TP]:\n{cm}")

if __name__ == "__main__":
    main()