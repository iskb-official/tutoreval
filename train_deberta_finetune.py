import pandas as pd
import numpy as np
import torch
from datasets import Dataset, ClassLabel
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
import evaluate

def load_and_prep_data(filepath):
    print("Loading data and generating MRBench labels...")
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
    
    # Convert to Hugging Face Dataset format
    return Dataset.from_pandas(df[['response', 'label']])

def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    recall_metric = evaluate.load("recall")
    precision_metric = evaluate.load("precision")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    recall = recall_metric.compute(predictions=predictions, references=labels)["recall"]
    precision = precision_metric.compute(predictions=predictions, references=labels)["precision"]
    
    return {"accuracy": acc, "f1": f1, "recall": recall, "precision": precision}

def main():
    MODEL_NAME = "microsoft/deberta-v3-small"
    DATA_PATH = "MRBench_V2_flat.csv"
    
    # 1. Load Data
    dataset = load_and_prep_data(DATA_PATH)
    
    # NEW LINE: explicitly cast the label column to ClassLabel
    dataset = dataset.cast_column("label", ClassLabel(num_classes=2, names=["POOR", "GOOD"]))
    
    # Split into 80% train, 20% test
    dataset = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
    
    # 2. Tokenization
    print("Downloading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):
        return tokenizer(examples["response"], truncation=True, max_length=256)
        
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 3. Load Model
    print("Downloading Model Architecture...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2,
        id2label={0: "POOR", 1: "GOOD"},
        label2id={"POOR": 0, "GOOD": 1}
    )
    
# 4. Define Training Arguments
    training_args = TrainingArguments(
        output_dir="./deberta_mrbench_model",
        eval_strategy="epoch",  # <--- Change this line right here
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    
    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer, # <--- Updated this line
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 6. Train!
    print("Starting Deep Fine-Tuning...")
    trainer.train()
    
    # 7. Final Evaluation
    print("\n=== FINAL TEST SET EVALUATION ===")
    results = trainer.evaluate()
    print(f"Accuracy: {results['eval_accuracy']:.4f}")
    print(f"Recall (Good): {results['eval_recall']:.4f}")
    print(f"F1 Score: {results['eval_f1']:.4f}")
    
    # Save for your inference script
    trainer.save_model("./deberta_mrbench_final")
    print("Model saved to ./deberta_mrbench_final. Ready for edge deployment.")

if __name__ == "__main__":
    main()