# Robust Detection of Human Life Threats on Social Network Platforms using Domain-Adaptive Transformers

This repository contains the official implementation of our paper:

**"ROBUST DETECTION OF HUMAN LIFE THREATS ON SOCIAL NETWORK PLATFORMS USING DOMAIN-ADAPTIVE TRANSFORMERS"**

The project leverages **BERT-based transformers with domain adaptation** and **focal loss** to address data imbalance, providing robust classification of life-threatening content into multiple classes.

---

## 📌 Features
- Preprocessing and cleaning of threat-related datasets  
- Custom **Focal Loss** implementation for handling class imbalance  
- Domain-adaptive fine-tuning with **Hugging Face Transformers**  
- Automated training and evaluation pipeline with **custom Trainer**  
- Metrics: Accuracy, Precision, Recall, F1 (Macro & Weighted)  
- Results export (JSON, Excel, PNG) for reproducibility  
- Visualizations: Training curves, Confusion Matrix, Class Distribution  

---

## 🛠️ Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt



The Proposed Dataset Multi-class Human Life Threat Dataset can be downloaded from google drive link. (Once paper is published)
