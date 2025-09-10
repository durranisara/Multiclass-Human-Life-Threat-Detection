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


# Custom Focal Loss implementation
class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = torch.nn.functional.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=inputs.size(-1)).float()
        # Expand weights to match batch size, indexed by targets
        weights = self.weight.to(inputs.device) if self.weight is not None else None
        if weights is not None:
            weights = weights[targets]  # Shape: [batch_size]

        focal_weight = ((1 - probs) ** self.gamma) * targets_one_hot
        loss = -focal_weight * log_probs

        if weights is not None:
            loss = loss * weights.unsqueeze(-1)  # Broadcast weights across class dimension

        if self.reduction == 'mean':
            return loss.sum() / (targets_one_hot.sum() + 1e-8)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


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

# Compute class weights for imbalance handling
class_counts = Counter(train_df['label'])
num_classes = len(class_counts)
total_samples = len(train_df)
class_weights = [total_samples / (num_classes * class_counts[i]) if class_counts[i] > 0 else 1.0 for i in
                 range(num_classes)]
class_weights = torch.tensor(class_weights, dtype=torch.float)
print("Class weights (for handling imbalance):", class_weights)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df[['Comment Text', 'label']].rename(columns={'Comment Text': 'text'}))
test_dataset = Dataset.from_pandas(test_df[['Comment Text', 'label']].rename(columns={'Comment Text': 'text'}))

# Step 2: Tokenization
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


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
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

# Move model and weights to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
class_weights = class_weights.to(device)


# Custom Trainer with Focal Loss
class WeightedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(WeightedTrainer, self).__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(weight=class_weights, gamma=2.0, reduction='mean')

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").to(model.device)
        outputs = model(**{k: v.to(model.device) for k, v in inputs.items()})
        logits = outputs.logits
        loss = self.focal_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss


# Training Arguments - UPDATED PATHS
training_args = TrainingArguments(
    output_dir='./bert_results',  # Changed to bert_results
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./bert_logs',  # Changed to bert_logs
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
trainer = WeightedTrainer(
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
    class_weights = class_weights.to("cpu")
    # Reinitialize trainer with CPU model
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    train_result = trainer.train()
    train_metrics = train_result.metrics
    train_history = trainer.state.log_history

# Save training results - UPDATED PATH
os.makedirs('bert_results', exist_ok=True)
with open('bert_results/train_results.json', 'w') as f:
    json.dump({'metrics': train_metrics, 'history': train_history, 'class_weights': class_weights.tolist()}, f,
              indent=4)
print("Training results saved to bert_results/train_results.json")

# Step 5: Evaluate on Test Set
test_predictions = trainer.predict(test_dataset)
test_metrics = test_predictions.metrics
test_preds = np.argmax(test_predictions.predictions, axis=-1)
test_labels = test_predictions.label_ids

# Detailed classification report
class_report = classification_report(test_labels, test_preds, target_names=list(label_map.keys()), zero_division=0)
print("BERT Test Classification Report:")
print(class_report)

# Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)

# Save test results to CSV - UPDATED PATH
test_results_df = pd.DataFrame({
    'True_Label': test_labels,
    'Predicted_Label': test_preds,
    'Class': [list(label_map.keys())[l] for l in test_labels],
    'Predicted_Class': [list(label_map.keys())[p] for p in test_preds]
})
test_results_df['Accuracy'] = test_results_df['True_Label'] == test_results_df['Predicted_Label']
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision_Macro', 'Recall_Macro', 'F1_Macro', 'Precision_Weighted', 'Recall_Weighted',
               'F1_Weighted'],
    'Value': [test_metrics['test_accuracy'], test_metrics['test_precision_macro'], test_metrics['test_recall_macro'],
              test_metrics['test_f1_macro'], test_metrics['test_precision_weighted'],
              test_metrics['test_recall_weighted'],
              test_metrics['test_f1_weighted']]
})
class_report_df = pd.DataFrame(
    classification_report(test_labels, test_preds, target_names=list(label_map.keys()), output_dict=True)).T
cm_df = pd.DataFrame(cm, index=list(label_map.keys()), columns=list(label_map.keys()))

with pd.ExcelWriter('bert_results/test_results.xlsx') as writer:  # Changed path
    test_results_df.to_excel(writer, sheet_name='Predictions', index=False)
    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
    class_report_df.to_excel(writer, sheet_name='Class_Report', index=True)
    cm_df.to_excel(writer, sheet_name='Confusion_Matrix', index=True)

print("Test results saved to bert_results/test_results.xlsx")

# Step 6: Graphs
# Plot 1: Training Loss Curve
epochs = [log['epoch'] for log in train_history if 'loss' in log]
losses = [log['loss'] for log in train_history if 'loss' in log]
plt.figure(figsize=(8, 5))
plt.plot(epochs, losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('BERT Training Loss Curve')  # Updated title
plt.legend()
plt.savefig('bert_results/training_loss_curve.png')  # Updated path
plt.show()

# Plot 2: Confusion Matrix Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(label_map.keys()),
            yticklabels=list(label_map.keys()))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('BERT Confusion Matrix')  # Updated title
plt.savefig('bert_results/confusion_matrix.png')  # Updated path
plt.show()

# Plot 3: Class Distribution Bar Chart
class_counts_list = [class_counts[i] for i in range(num_classes)]
plt.figure(figsize=(8, 5))
plt.bar(list(label_map.keys()), class_counts_list, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Class Distribution in Training Set (BERT)')  # Updated title
plt.savefig('bert_results/class_distribution.png')  # Updated path
plt.show()

print("Graphs saved as PNG files in bert_results folder.")  # Updated message