import pandas as pd
import numpy as np
import scipy.sparse as sp
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_and_prep_data(filepath):
    print("Loading dataset and generating binary labels...")
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['response'])
    
    def determine_label(row):
        guidance = str(row.get('Providing_Guidance', '')).strip()
        actionability = str(row.get('Actionability', '')).strip()
        coherence = str(row.get('Coherence', '')).strip()
        if guidance in ['Yes', 'To some extent'] and actionability in ['Yes', 'To some extent'] and coherence == 'Yes':
            return 1
        return 0

    df['label'] = df.apply(determine_label, axis=1)
    return df['response'].tolist(), df['label'].tolist()

def extract_combined_features(texts):
    print("1. Extracting TF-IDF features (Lexical)...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=2)
    X_tfidf = tfidf.fit_transform(texts)
    
    print("2. Extracting Semantic Embeddings (Contextual)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    X_dense = model.encode(texts, show_progress_bar=True)
    
    print("3. Concatenating Features...")
    # Combine sparse TF-IDF with dense embeddings into one massive feature space
    X_combined = sp.hstack([X_tfidf, X_dense], format='csr')
    
    return X_combined, tfidf, model

def apply_pedagogical_rules(texts, base_probs):
    print("Applying V4 Rule-Based Boosting...")
    final_preds = []
    
    # The exact canonical phrases from your V4 methodology
    scaffolding_phrases = ["good try", "nice effort", "nice job", "well done", 
                           "step by step", "let us think", "try using", 
                           "remember that", "notice that", "good question", 
                           "good start", "check again"]
    
    for i, text in enumerate(texts):
        text_lower = text.lower()
        p_ml = base_probs[i][1] # Probability of being GOOD
        
        # Count scaffolding phrases
        c = sum(1 for phrase in scaffolding_phrases if phrase in text_lower)
        
        # Apply the boost formula: min(0.15 * c, 0.5)
        boost = min(0.15 * c, 0.5)
        p_final = min(p_ml + boost, 0.95)
        
        # Final decision boundary
        final_preds.append(1 if p_final >= 0.5 else 0)
        
    return final_preds

def main():
    DATA_PATH = "MRBench_V2_flat.csv"
    texts, labels = load_and_prep_data(DATA_PATH)
    
    X_combined, tfidf, encoder = extract_combined_features(texts)
    y = np.array(labels)
    
    # We must keep the texts aligned with the test set to apply the rule engine later
    X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
        X_combined, y, texts, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training Ultimate XGBoost...")
    # Using the optimized parameters we found earlier
    scale_weight = list(y_train).count(0) / list(y_train).count(1)
    xgb_model = XGBClassifier(
        max_depth=7, 
        learning_rate=0.2, 
        n_estimators=200, 
        subsample=1.0,
        scale_pos_weight=scale_weight,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    
    # Get base ML probabilities
    base_probs = xgb_model.predict_proba(X_test)
    
    # Apply the Pedagogical Rule Engine
    final_predictions = apply_pedagogical_rules(texts_test, base_probs)
    
    print("\n=== FINAL HYBRID EVALUATION ===")
    print(f"Final Accuracy: {accuracy_score(y_test, final_predictions):.4f}")
    print(classification_report(y_test, final_predictions, target_names=['Poor', 'Good']))
    
    joblib.dump(xgb_model, 'ultimate_xgb_model.pkl')
    joblib.dump(tfidf, 'ultimate_tfidf.pkl')

if __name__ == "__main__":
    main()