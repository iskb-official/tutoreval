import pandas as pd
import numpy as np
import scipy.sparse as sp
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
import xgboost as xgb
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

def main():
    print("Loading data for Out-of-Fold (OOF) True Kappa calculation...")
    df = pd.read_csv("MRBench_V2_flat.csv")
    df = df.dropna(subset=['response', 'conversation_id'])
    
    def determine_label(row):
        g = str(row.get('Providing_Guidance', '')).strip()
        a = str(row.get('Actionability', '')).strip()
        c = str(row.get('Coherence', '')).strip()
        if g in ['Yes', 'To some extent'] and a in ['Yes', 'To some extent'] and c == 'Yes':
            return 1
        return 0

    df['ground_truth'] = df.apply(determine_label, axis=1)
    # FIX: Convert pandas column/array to a standard list of python strings
    texts = df['response'].astype(str).tolist()
    y = df['ground_truth'].values
    groups = df['conversation_id'].values
    
    # Extract TF-IDF
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')
    X_tfidf = tfidf.fit_transform(texts)
    
    # Extract Embeddings
    print("Encoding texts...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    X_dense = encoder.encode(texts, show_progress_bar=True)
    
    X_combined = sp.hstack([X_tfidf, X_dense], format='csr')
    
    # Stratified Group K-Fold OOF Predictions
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_preds = np.zeros(len(y))
    
    scaffolding_phrases = ["good try", "nice effort", "nice job", "well done", 
                           "step by step", "let us think", "try using", 
                           "remember that", "notice that", "good question", 
                           "good start", "check again"]
    
    print("Running Stratified Group 5-Fold Cross-Validation for OOF evaluation...")
    for train_idx, test_idx in sgkf.split(X_combined, y, groups):
        X_train, X_test = X_combined.tocsr()[train_idx], X_combined.tocsr()[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, scale_pos_weight=1.4, random_state=42)
        model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_test)[:, 1]
        
        # Apply Rule Boost
        fold_preds = []
        for i, idx in enumerate(test_idx):
            text_lower = texts[idx].lower()
            p_ml = probs[i]
            c = sum(1 for phrase in scaffolding_phrases if phrase in text_lower)
            boost = min(0.15 * c, 0.5)
            p_final = min(p_ml + boost, 0.95)
            fold_preds.append(1 if p_final >= 0.5 else 0)
            
        oof_preds[test_idx] = fold_preds
        
    # Calculate True OOF Kappa
    oof_kappa = cohen_kappa_score(y, oof_preds)
    oof_acc = accuracy_score(y, oof_preds)
    cm = confusion_matrix(y, oof_preds)
    
    print("\n==================================================")
    print(" TRUE OUT-OF-FOLD (OOF) HYBRID EVALUATION METRICS ")
    print("==================================================")
    print(f"OOF Accuracy: {oof_acc * 100:.2f}%")
    print(f"OOF Cohen's Kappa (κ): {oof_kappa:.3f}")
    print(f"OOF Confusion Matrix:\n{cm}")

if __name__ == "__main__":
    main()