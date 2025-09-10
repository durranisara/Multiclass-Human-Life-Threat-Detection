import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import Counter
import os

# Set CUDA debug flags for better error reporting
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Step 1: Load and Preprocess Data
df = pd.read_csv('D:/threat dataset/threat_dataset.csv', encoding='windows-1252')
print("Dataset loaded. Sample:")
print(df.head())

# Validate dataset
df = df.dropna(subset=['Comment Text', 'Label'])
df = df[df['Comment Text'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]
print(f"After cleaning, dataset size: {len(df)}")

# Map labels to integers
label_map = {
    'non-threat': 0,
    'weak threat': 1,
    'moderate threat': 2,
    'strong threat': 3,
    'public threat': 4
}
df['label'] = df['Label'].map(label_map)
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

# Split into train and test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df[['Comment Text', 'label']].rename(columns={'Comment Text': 'text'}))
test_dataset = Dataset.from_pandas(test_df[['Comment Text', 'label']].rename(columns={'Comment Text': 'text'}))

# Step 2: Tokenization
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

def tokenize_function(examples):
    texts = [str(text) if isinstance(text, str) else "" for text in examples['text']]
    return tokenizer(texts, padding='max_length', truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Step 3: Model Setup
model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=5)

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training Arguments
training_args = TrainingArguments(
    output_dir='./roberta_results',
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./roberta_logs',
    logging_steps=10,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy'
)

# Custom metric computation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    return {
        'accuracy': acc,
        'precision_macro': report['macro avg']['precision'],
        'recall_macro': report['macro avg']['recall'],
        'f1_macro': report['macro avg']['f1-score'],
        'precision_weighted': report['weighted avg']['precision'],
        'recall_weighted': report['weighted avg']['recall'],
        'f1_weighted': report['weighted avg']['f1-score']
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Step 4: Train the Model
try:
    train_result = trainer.train()
    train_metrics = train_result.metrics
    train_history = trainer.state.log_history
except RuntimeError as e:
    print(f"Training failed on GPU: {e}")
    print("Retrying on CPU...")
    model.to("cpu")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    train_result = trainer.train()
    train_metrics = train_result.metrics
    train_history = trainer.state.log_history

# Save training results
with open('roberta_results/train_results.json', 'w') as f:
    json.dump({'metrics': train_metrics, 'history': train_history}, f, indent=4)
print("Training results saved to roberta_results/train_results.json")

# Step 5: Evaluate on Test Set
test_predictions = trainer.predict(test_dataset)
test_metrics = test_predictions.metrics
test_preds = np.argmax(test_predictions.predictions, axis=-1)
test_labels = test_predictions.label_ids

# Detailed classification report
class_report = classification_report(test_labels, test_preds, target_names=list(label_map.keys()), zero_division=0)
print("RoBERTa Test Classification Report:")
print(class_report)

# Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)

# Save test results to CSV
test_results_df = pd.DataFrame({
    'True_Label': test_labels,
    'Predicted_Label': test_preds,
    'Class': [list(label_map.keys())[l] for l in test_labels],
    'Predicted_Class': [list(label_map.keys())[p] for p in test_preds]
})
test_results_df['Accuracy'] = test_results_df['True_Label'] == test_results_df['Predicted_Label']
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision_Macro', 'Recall_Macro', 'F1_Macro', 'Precision_Weighted', 'Recall_Weighted', 'F1_Weighted'],
    'Value': [test_metrics['test_accuracy'], test_metrics['test_precision_macro'], test_metrics['test_recall_macro'],
              test_metrics['test_f1_macro'], test_metrics['test_precision_weighted'], test_metrics['test_recall_weighted'],
              test_metrics['test_f1_weighted']]
})
class_report_df = pd.DataFrame(classification_report(test_labels, test_preds, target_names=list(label_map.keys()), output_dict=True)).T
cm_df = pd.DataFrame(cm, index=list(label_map.keys()), columns=list(label_map.keys()))

with pd.ExcelWriter('roberta_results/test_results.xlsx') as writer:
    test_results_df.to_excel(writer, sheet_name='Predictions', index=False)
    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
    class_report_df.to_excel(writer, sheet_name='Class_Report', index=True)
    cm_df.to_excel(writer, sheet_name='Confusion_Matrix', index=True)

print("Test results saved to roberta_results/test_results.xlsx")

# Step 6: Graphs
# Compute class counts for the bar chart
class_counts = Counter(train_df['label'])
class_counts_list = [class_counts[i] for i in range(len(label_map))]

# Plot 1: Training Loss Curve
epochs = [log['epoch'] for log in train_history if 'loss' in log]
losses = [log['loss'] for log in train_history if 'loss' in log]
plt.figure(figsize=(8, 5))
plt.plot(epochs, losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('RoBERTa Training Loss Curve')
plt.legend()
plt.savefig('roberta_results/training_loss_curve.png')
plt.show()

# Plot 2: Confusion Matrix Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(label_map.keys()), yticklabels=list(label_map.keys()))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('RoBERTa Confusion Matrix')
plt.savefig('roberta_results/confusion_matrix.png')
plt.show()

# Plot 3: Class Distribution Bar Chart
plt.figure(figsize=(8, 5))
plt.bar(list(label_map.keys()), class_counts_list, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Class Distribution in Training Set (RoBERTa)')
plt.savefig('roberta_results/class_distribution.png')
plt.show()

print("Graphs saved as PNG files in roberta_results folder.")