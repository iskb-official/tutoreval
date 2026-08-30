import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset, ClassLabel
import evaluate
import warnings
import os

warnings.filterwarnings('ignore')

# --- 1. DATA PREPARATION ---
def load_and_prep_data(filepath):
    print("Loading dataset and generating binary labels...")
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['response', 'conversation_id'])
    
    def determine_label(row):
        guidance = str(row.get('Providing_Guidance', '')).strip()
        actionability = str(row.get('Actionability', '')).strip()
        coherence = str(row.get('Coherence', '')).strip()
        if guidance in ['Yes', 'To some extent'] and actionability in ['Yes', 'To some extent'] and coherence == 'Yes':
            return 1
        return 0

    df['label'] = df.apply(determine_label, axis=1)
    return df

# --- 2. RULE-ONLY BASELINE ---
def evaluate_rule_only(df):
    print("\n--- Evaluating Rule-Only Baseline ---")
    scaffolding_phrases = [
        "good try", "nice effort", "nice job", "well done", "step by step", 
        "let us think", "try using", "remember that", "notice that", 
        "good question", "good start", "check again", "let's go", "try to", 
        "almost", "almost there", "not quite", "check whether", "what do you", 
        "how about", "let me help"
    ]
    
    y_true = df['label'].values
    y_pred = []
    
    for text in df['response']:
        text_lower = str(text).lower()
        if any(phrase in text_lower for phrase in scaffolding_phrases):
            y_pred.append(1)
        else:
            y_pred.append(0)
            
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    
    print(f"Rule-Only -> Accuracy: {acc:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
    return acc, f1, auc, y_true, y_pred

# --- 3. LOGISTIC REGRESSION BASELINE ---
def evaluate_logistic_regression(df):
    print("\n--- Evaluating TF-IDF + Logistic Regression Baseline ---")
    
    X = df['response'].values
    y = df['label'].values
    groups = df['conversation_id'].values
    
    sgkf = StratifiedGroupKFold(n_splits=5)
    
    acc_scores, f1_scores, auc_scores = [], [], []
    y_true_all, y_prob_all = [], []
    
    for train_idx, test_idx in sgkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=2)
        X_train_vec = tfidf.fit_transform(X_train)
        X_test_vec = tfidf.transform(X_test)
        
        # Balance class weights
        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        model.fit(X_train_vec, y_train)
        
        y_pred = model.predict(X_test_vec)
        y_prob = model.predict_proba(X_test_vec)[:, 1]
        
        acc_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        auc_scores.append(roc_auc_score(y_test, y_prob))
        
        y_true_all.extend(y_test)
        y_prob_all.extend(y_prob)

    acc = np.mean(acc_scores)
    f1 = np.mean(f1_scores)
    auc = np.mean(auc_scores)
    
    print(f"Logistic Regression -> Mean Accuracy: {acc:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
    return acc, f1, auc, y_true_all, y_prob_all

# --- 4. DISTILBERT BASELINE ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {"accuracy": acc, "f1": f1}

def evaluate_distilbert(df):
    print("\n--- Evaluating DistilBERT Baseline ---")
    
    # We will use a standard 80/20 split grouped by conversation_id to save compute time
    # while still preventing data leakage.
    groups = df['conversation_id'].unique()
    np.random.seed(42)
    np.random.shuffle(groups)
    
    split_idx = int(len(groups) * 0.8)
    train_groups = groups[:split_idx]
    
    train_df = df[df['conversation_id'].isin(train_groups)]
    test_df = df[~df['conversation_id'].isin(train_groups)]
    
    train_dataset = Dataset.from_pandas(train_df[['response', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['response', 'label']])
    
    train_dataset = train_dataset.cast_column("label", ClassLabel(num_classes=2, names=["POOR", "GOOD"]))
    test_dataset = test_dataset.cast_column("label", ClassLabel(num_classes=2, names=["POOR", "GOOD"]))
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_fn(examples):
        return tokenizer(examples["response"], truncation=True, max_length=256)
        
    train_tokenized = train_dataset.map(tokenize_fn, batched=True)
    test_tokenized = test_dataset.map(tokenize_fn, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label={0: "POOR", 1: "GOOD"}, label2id={"POOR": 0, "GOOD": 1}
    )
    
    training_args = TrainingArguments(
        output_dir="./distilbert_temp",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("Training DistilBERT (this will take time on CPU)...")
    trainer.train()
    
    results = trainer.evaluate()
    
    # Generate probabilities for AUC
    raw_preds = trainer.predict(test_tokenized)
    probs = torch.nn.functional.softmax(torch.tensor(raw_preds.predictions), dim=-1)[:, 1].numpy()
    true_labels = raw_preds.label_ids
    
    acc = results['eval_accuracy']
    f1 = results['eval_f1']
    auc = roc_auc_score(true_labels, probs)
    
    print(f"DistilBERT -> Accuracy: {acc:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
    return acc, f1, auc, true_labels, probs

# --- 5. VISUALIZATION AND REPORTING ---
def plot_results(metrics_dict):
    print("\n--- Generating 500 DPI PDF Figures ---")
    
    # 1. Bar Chart Comparison
    df_plot = pd.DataFrame(metrics_dict).T.reset_index()
    df_plot.columns = ['Model', 'Accuracy', 'F1-Score', 'AUC']
    df_melt = pd.melt(df_plot, id_vars=['Model'], var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Score', hue='Metric', data=df_melt, palette='viridis')
    plt.title('Performance Comparison of Pedagogical Evaluation Models', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.ylabel('Score')
    plt.legend(loc='lower right')
    
    bar_chart_path = 'baseline_metrics_comparison.pdf'
    plt.savefig(bar_chart_path, format='pdf', dpi=500, bbox_inches='tight')
    print(f"Saved bar chart to {bar_chart_path}")
    plt.close()

def main():
    DATA_PATH = "MRBench_V2_flat.csv"
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find {DATA_PATH}. Please ensure it is in the same directory.")
        
    df = load_and_prep_data(DATA_PATH)
    
    # Run Baselines
    rule_acc, rule_f1, rule_auc, rule_y, rule_pred = evaluate_rule_only(df)
    lr_acc, lr_f1, lr_auc, lr_y, lr_prob = evaluate_logistic_regression(df)
    db_acc, db_f1, db_auc, db_y, db_prob = evaluate_distilbert(df)
    
    # Store metrics (Including your current XGBoost metrics for the chart)
    metrics = {
        'Rule-Only': {'Accuracy': rule_acc, 'F1-Score': rule_f1, 'AUC': rule_auc},
        'TF-IDF + LogReg': {'Accuracy': lr_acc, 'F1-Score': lr_f1, 'AUC': lr_auc},
        'DistilBERT': {'Accuracy': db_acc, 'F1-Score': db_f1, 'AUC': db_auc},
        # From your manuscript draft
        'TF-IDF + XGBoost (Proposed)': {'Accuracy': 0.687, 'F1-Score': 0.740, 'AUC': 0.712}
    }
    
    # Create PDF charts
    plot_results(metrics)
    print("\nAll baseline evaluations complete. Ready for manuscript integration.")

if __name__ == "__main__":
    main()