import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_and_prep_data(filepath):
    print("Loading dataset and generating binary labels from MRBench taxonomy...")
    df = pd.read_csv(filepath)
    
    # 1. Drop rows where the text response is missing
    df = df.dropna(subset=['response'])
    
    # 2. Define the taxonomy logic from the paper
    def determine_label(row):
        guidance = str(row.get('Providing_Guidance', '')).strip()
        actionability = str(row.get('Actionability', '')).strip()
        coherence = str(row.get('Coherence', '')).strip()
        
        guidance_ok = guidance in ['Yes', 'To some extent']
        actionability_ok = actionability in ['Yes', 'To some extent']
        coherence_ok = coherence == 'Yes'
        
        if guidance_ok and actionability_ok and coherence_ok:
            return 1 # GOOD
        else:
            return 0 # POOR

    # 3. Apply the logic to create the 'label' column
    df['label'] = df.apply(determine_label, axis=1)
    
    print(f"Generated Labels: {df['label'].value_counts().to_dict()}")
    
    return df['response'].tolist(), df['label'].tolist()

def generate_embeddings(texts):
    print("Generating semantic embeddings (this may take a moment)...")
    # all-MiniLM-L6-v2 is highly accurate but optimized for edge/low-compute environments
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def optimize_and_train_xgb(X_train, y_train):
    print("Optimizing XGBoost parameters...")
    
    # Calculate scale_pos_weight to handle Good/Poor class imbalance
    num_neg = y_train.count(0)
    num_pos = y_train.count(1)
    scale_weight = num_neg / num_pos if num_pos > 0 else 1

    xgb_model = XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_weight,
        eval_metric='logloss',
        use_label_encoder=False
    )

    # Grid search for hyperparameter tuning
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    print(f"Best parameters found: {grid_search.best_params_}")
    
    return grid_search.best_estimator_

def main():
    # 1. Setup paths (Update this to your actual training data path)
    DATA_PATH = "MRBench_V2_flat.csv"
    
    # 2. Load Data
    texts, labels = load_and_prep_data(DATA_PATH)
    
    # 3. Extract Features
    X = generate_embeddings(texts)
    y = np.array(labels)
    
    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Train Model
    best_xgb = optimize_and_train_xgb(X_train, list(y_train))
    
    # 6. Evaluate Accuracy
    print("\n--- Evaluation on Test Set ---")
    predictions = best_xgb.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions, target_names=['Poor', 'Good']))
    
    # 7. Save the optimized models for the inference engine
    print("Saving models to disk...")
    joblib.dump(best_xgb, 'optimized_xgb_model.pkl')
    print("Training complete. Pipeline ready for inference.")

if __name__ == "__main__":
    main()