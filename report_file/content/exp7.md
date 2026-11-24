# Experiment 7: IMDB Movie Review Sentiment Analysis

## Abstract
This experiment builds a binary sentiment classifier for the IMDB Movie Reviews dataset (25,000 positive and 25,000 negative reviews) and compares four modelling families on exactly the same train/validation/test split. A TF-IDF + Logistic Regression baseline is compared with three neural models trained from scratch or fine-tuned: a bidirectional LSTM, a vanilla Transformer encoder, and the pretrained BERT model (`bert-base-uncased`). All reviews are cleaned, lower-cased and stripped of HTML tags and punctuation before tokenization; a 25,000-word vocabulary with right-padded sequences of length 200 is used for the from-scratch networks, while BERT uses its own WordPiece tokenizer with a length of 128. Accuracy, macro-averaged precision, recall and F1, plus a confusion matrix and a classification report, are computed for each model on the 10,000-review held-out test set. Logistic Regression achieves the best macro F1 (0.9091, 11.9 s), followed by BERT (0.8970, 3119.6 s), the LSTM (0.8875, 254.0 s) and the vanilla Transformer (0.8611, 282.3 s). The classical baseline is therefore both the most accurate and roughly 260 times faster than BERT on this data; an error-overlap analysis also shows that the four models make substantially different mistakes.

## Description
The notebook `EXP-7/exp-7.ipynb` begins by setting all random seeds to 42, selecting the MPS device when available, and defining a single configuration dictionary (`CFG`) with the vocabulary size, sequence lengths, batch sizes, epoch counts and learning rates. The dataset file `IMDB-Dataset.csv` is loaded with Pandas, its shape, class balance and missing values are inspected, and the class distribution, review-length histogram and character-length box plot are drawn. Two optional environment flags are supported: `EXP7_QUICK=1` performs a fast smoke test on small subsets, and `EXP7_RESUME=1` reloads cached predictions from `results/` instead of retraining.

The text cleanup function removes HTML tags, lowercases, removes non-alphabetic characters and collapses whitespace. The labels are encoded as 1 for positive and 0 for negative. A stratified 80/20 train-test split is followed by a stratified 90/10 split of the training portion into training and validation sets. A regex tokenizer extracts words, the 25,000 most common training words form the vocabulary (with `<pad>` and `<unk>` reserved), and each review is encoded to a fixed-length integer sequence. PyTorch `Dataset` and `DataLoader` wrappers feed batches to the neural networks. This yields 36,000 training, 4,000 validation and 10,000 test reviews shared by all four models.

A common `train_classifier` loop performs training with cross-entropy loss and Adam, evaluates the validation loss after each epoch, keeps the best weights, and records epoch times; `predict` generates test probabilities in batches. The baseline vectorizes text with word and character n-gram TF-IDF (30,000 features, `min_df=2`, sublinear term frequency) and fits an L2 Logistic Regression solver. The LSTM uses a 128-dimensional embedding, two bidirectional LSTM layers with a hidden size of 128, masked mean pooling and a dropout-regularized output layer. The vanilla Transformer uses learned positional embeddings, three `nn.TransformerEncoderLayer` blocks with four attention heads and a 256-unit feed-forward layer. BERT is loaded from the Hugging Face `transformers` library with a two-label sequence-classification head, and fine-tuned for one epoch at learning rate 3e-5 with gradient clipping and a linear warm-up schedule; `USE_TF=0` is set before importing `transformers` to avoid a TensorFlow import conflict in this environment.

Every model's predictions are scored with the same `compute_metrics` helper, which reports accuracy, macro precision, recall and F1 and stores train time. The results are written to `results/comparison.csv`, visualized as a grouped bar chart (`exp7_comparison.png`) and accompanied by four labelled confusion matrices (`exp7_confusion_matrices.png`). Training-loss curves for the three neural models and an error-overlap analysis (reviews misclassified by all models versus only one model) complete the comparison.

## Code
### Shared setup, seeds and configuration
```python
import os
os.environ["USE_TF"] = "0"  # transformers must not import TensorFlow here

import random, re, time, warnings
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DATASET_PATH = "./IMDB-Dataset.csv"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CFG = dict(
    vocab_size=25000, max_len=200, max_len_bert=128,
    embed_dim=128, batch=64, batch_bert=32,
    epochs_lstm=5, epochs_trf=3, epochs_bert=1,
    lr=1e-3, lr_bert=3e-5, tfidf_features=30000,
)
ALL_METRICS = {}
```

### Cleaning, stratified splitting and vocabulary encoding
```python
def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)     # drop HTML tags
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)    # letters + spaces only
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df = pd.read_csv(DATASET_PATH)
df["clean"] = df["review"].map(clean_text)
y = (df["sentiment"] == "positive").astype(int).to_numpy()
idx = np.arange(len(df))

train_val_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1,
                                      random_state=SEED, stratify=y[train_val_idx])

TOKEN_RE = re.compile(r"[a-z']+")
def tokenize(text):
    return TOKEN_RE.findall(text)

counter = Counter()
for t in X_train_txt:
    counter.update(tokenize(t))
vocab = {"<pad>": 0, "<unk>": 1}
for word, _ in counter.most_common(CFG["vocab_size"] - 2):
    vocab[word] = len(vocab)

def encode(texts, max_len):
    arr = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(texts):
        for j, w in enumerate(tokenize(t)[:max_len]):
            arr[i, j] = vocab.get(w, 1)
    return arr
```

### Neural architectures (LSTM, vanilla Transformer, BERT head)
```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden=128, num_layers=2,
                 num_classes=2, pad_idx=0, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=num_layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, ids):
        mask = ids.ne(0)
        out, _ = self.lstm(self.dropout(self.embedding(ids)))
        m = mask.unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.fc(self.dropout(pooled))

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=128, nhead=4, num_layers=3,
                 dim_ff=256, dropout=0.1, num_classes=2, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.position = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=dim_ff, dropout=dropout,
                                           activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, ids):
        mask = ids.ne(0)
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        h = self.dropout(self.embedding(ids) + self.position(pos))
        h = self.encoder(h, src_key_padding_mask=~mask)
        h = self.norm(h[:, 0])
        return self.fc(self.dropout(h))

from transformers import AutoModelForSequenceClassification, AutoTokenizer
BERT_NAME = "bert-base-uncased"
bert_tok = AutoTokenizer.from_pretrained(BERT_NAME)

class BertClassifier(nn.Module):
    def __init__(self, name=BERT_NAME):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)

    def forward(self, ids):
        return self.bert(input_ids=ids, attention_mask=ids.ne(0)).logits
```

### Metrics, result table and comparison charts
```python
def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

def evaluate_model(name, y_true, y_pred, train_time=None):
    m = compute_metrics(y_true, y_pred)
    if train_time is not None:
        m["TrainTime(s)"] = round(float(train_time), 1)
    ALL_METRICS[name] = m
    print(f"===== {name} ===== Accuracy {m['Accuracy']:.4f} | F1 {m['F1']:.4f}")

cmp_df = (pd.DataFrame.from_dict(ALL_METRICS, orient="index")
          .rename_axis("Model")[["Accuracy", "Precision", "Recall", "F1", "TrainTime(s)"]])
cmp_df.round(4).to_csv(f"{RESULTS_DIR}/comparison.csv")

preds = [("Logistic Regression", y_pred_logreg), ("LSTM", y_pred_lstm),
         ("Vanilla Transformer", y_pred_trf), ("BERT", y_pred_bert)]
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, (name, p) in zip(axes.ravel(), preds):
    cm = confusion_matrix(y_test, p)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["negative", "positive"],
                yticklabels=["negative", "positive"])
    ax.set_title(f"{name}\nacc {accuracy_score(y_test, p):.4f}  "
                 f"F1 {f1_score(y_test, p, average='macro'):.4f}")
plt.tight_layout()
plt.savefig("exp7_confusion_matrices.png", bbox_inches="tight")
plt.show()
```

## Tables
### Model configurations
| Model | Representation / architecture | Epochs | Sequence length | Learning rate |
| --- | --- | --- | --- | --- |
| Logistic Regression | word + character TF-IDF, 30,000 features, L2 solver (C = 1.0) | — | — | — |
| LSTM | 128-d embedding, 2 bidirectional LSTM layers (hidden 128), masked mean pooling | 5 | 200 | 1e-3 |
| Vanilla Transformer | 3 encoder blocks, 4 attention heads, d_model 128, feed-forward 256 | 3 | 200 | 1e-3 |
| BERT | `bert-base-uncased` + 2-label head (109,483,778 parameters) | 1 | 128 | 3e-5 |

### Test-set performance (10,000 reviews, macro-averaged)
| Model | Accuracy | Precision | Recall | F1 | Train time (s) |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9091 | 0.9092 | 0.9091 | 0.9091 | 11.9 |
| BERT | 0.8970 | 0.8974 | 0.8970 | 0.8970 | 3119.6 |
| LSTM | 0.8875 | 0.8875 | 0.8875 | 0.8875 | 254.0 |
| Vanilla Transformer | 0.8611 | 0.8611 | 0.8611 | 0.8611 | 282.3 |

### Confusion matrix entries (rows: actual negative/positive; columns: predicted negative/positive)
| Model | TN | FP | FN | TP |
| --- | --- | --- | --- | --- |
| Logistic Regression | 4509 | 491 | 418 | 4582 |
| LSTM | 4422 | 578 | 547 | 4453 |
| Vanilla Transformer | 4320 | 680 | 709 | 4291 |
| BERT | 4409 | 591 | 439 | 4561 |

### Error overlap on the test set
| Category | Count |
| --- | --- |
| Misclassified by all four models | 266 |
| Wrong only for Logistic Regression (BERT correct) | 504 |
| Wrong only for BERT (Logistic Regression correct) | 625 |

## Figures
| path | caption |
| --- | --- |
| /Users/kartiken/Downloads/CLNLP-LAB/EXP-7/exp7_comparison.png | Grouped bar chart of macro precision, recall and F1 for the four models. |
| /Users/kartiken/Downloads/CLNLP-LAB/EXP-7/exp7_confusion_matrices.png | Confusion matrices of all four models on the held-out IMDB test set. |
