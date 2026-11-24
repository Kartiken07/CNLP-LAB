# EXP-9: AG News Classification - End-to-End NLP System

## Abstract
This project builds an end-to-end classical NLP pipeline for classifying AG News articles into World, Sports, Business and Sci/Tech. The dataset contains 120,000 training articles and 7,600 test articles, with exactly 30,000 training examples per class and no missing or duplicate rows. Articles were cleaned with lowercasing, URL and HTML removal, stopword removal and Porter stemming, then represented with Bag-of-Words and TF-IDF unigram-plus-bigram features. Seven model configurations were trained and compared using cross-validated hyperparameter search, a stratified validation split and the held-out test set. LinearSVC with TF-IDF achieved the best test accuracy of 92.36% and weighted F1 of 0.9234, ahead of logistic regression with TF-IDF at 92.00% and F1 0.9199. The selected LinearSVC was probability-calibrated with sigmoid calibration, reaching a calibrated test accuracy of 0.9238. A reusable `predict(text)` interface and a Flask web application expose the predicted category, confidence and class probabilities. End-to-end prediction on 1,000 raw test articles reproduced 0.928 accuracy.

## Description
The work is spread across five notebooks in `EXP-9/`: `01_EDA.ipynb`, `02_Text Preprocessing.ipynb`, `03_Feature_Extraction.ipynb`, `04_Model_Training_Evaluation.ipynb` and `05_Model_Comparison_Prediction.ipynb`. Supporting modules are `src/preprocess.py` and `src/predict.py`, serialized artifacts are in `models/`, evaluation outputs and plots are in `reports/`, and the Flask frontend is in `app/`. The environment uses Python 3.11, scikit-learn 1.9.1, pandas, NumPy, SciPy, NLTK, joblib, Matplotlib, Seaborn, Flask and the Hugging Face `datasets` library. The project documentation `PROJECT_REPORT.md` records the five-member team division of labor from data exploration through web integration.

The exploratory notebook loads `sh0416/ag_news`, converts the splits to pandas and maps labels 1-4 to World, Sports, Business and Sci/Tech. It verified shapes of (120,000, 3) for training and (7,600, 3) for test, zero missing values, zero duplicate rows and zero duplicate combined texts. Mean article length was 37.85 words with a standard deviation of 10.09, a minimum of 8 and a maximum of 177 words. The most frequent token was "the" with 205,478 occurrences, followed by "to" with 120,746. Cleaning in the final notebooks converts literal backslash line breaks to spaces and applies Porter stemming; it produced zero empty texts across both splits. CountVectorizer with `min_df=3` produced a 120,000 x 25,393 Bag-of-Words matrix with 19.50 nonzeros per document, while TfidfVectorizer with unigrams, bigrams and `min_df=3` produced a 120,000 x 100,000 matrix with 28.43 nonzeros per document.

Model development used a stratified 90:10 split of the training data, giving 108,000 training and 12,000 validation rows, plus a stratified 30,000-row subsample for three-fold hyperparameter search. Multinomial Naive Bayes chose alpha 0.1 on TF-IDF and 0.3 on BoW; logistic regression chose C=5; LinearSVC chose C=0.3; Random Forest used chi2 selection of the top 20,000 features with 150 trees; HistGradientBoosting used TruncatedSVD with 200 components. The observed test accuracies were 0.9076, 0.9018, 0.9200, 0.9024, 0.9236, 0.8989 and 0.8904 respectively. LinearSVC also had the fastest useful training time among the tuned TF-IDF models at 2.4 seconds and a prediction cost of about 0.2 ms per 1,000 documents.

The final stage selected LinearSVC because it had the highest weighted F1 among TF-IDF candidates and it required sigmoid calibration for confidence scores. After calibration on the full 120,000-row training features, the saved model achieved 0.9238 test accuracy and the metadata records the uncalibrated test accuracy 0.9236, F1 0.9234, C=0.3 and 120,000 calibration training rows. Sample inference produced the correct category for four hand-written articles with confidences 0.97 for Business, 1.00 for Sports, 1.00 for Sci/Tech and 0.94 for World. Confusion matrices show that Sports is the easiest class, with per-class F1 0.9707 for the selected model. The Flask app exposes `/`, `/api/predict`, `/api/model` and `/api/health`, and the repository includes 21 automated tests.

## Code

### Dataset loading and exploratory checks
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

ds = load_dataset("sh0416/ag_news")
train_data, test_data = ds["train"], ds["test"]
train_df = train_data.to_pandas()
test_df = test_data.to_pandas()

print("Training Shape:", train_df.shape)
print("Testing Shape:", test_df.shape)

label_mapping = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
train_df["category"] = train_df["label"].map(label_mapping)
test_df["category"] = test_df["label"].map(label_mapping)

train_df["text"] = train_df["title"] + " " + train_df["description"]
test_df["text"] = test_df["title"] + " " + test_df["description"]

print(train_df["label"].value_counts())
print(train_df.isnull().sum())
print("Training duplicates:", train_df.duplicated().sum())
train_df["word_count"] = train_df["text"].apply(lambda x: len(x.split()))
print(train_df["word_count"].describe())

sns.countplot(data=train_df, x="category")
plt.title("Distribution of News Categories")
plt.show()
```

### Shared cleaning and feature extraction
```python
import re
from functools import lru_cache
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sp
import joblib

STOP_WORDS = frozenset(ENGLISH_STOP_WORDS)
_stemmer = PorterStemmer()
_URL = re.compile(r"http\S+|www\S+")
_HTML = re.compile(r"<[^>]+>")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    return _stemmer.stem(word)


def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().replace("\\", " ")
    text = _URL.sub(" ", text)
    text = _HTML.sub(" ", text)
    text = _NON_ALPHA.sub("", text)
    text = _SPACES.sub(" ", text).strip()
    return " ".join(_stem(w) for w in text.split() if w not in STOP_WORDS and len(w) > 1)


train = pd.read_csv("ag_news_train.csv")
test = pd.read_csv("ag_news_test.csv")
for df in (train, test):
    df["clean_text"] = df["text"].apply(clean_text)

le = LabelEncoder().fit(train["category"])
y_train, y_test = le.transform(train["category"]), le.transform(test["category"])
bow = CountVectorizer(max_features=50000, min_df=3)
Xb_train = bow.fit_transform(train.clean_text)
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=100000,
                        min_df=3, sublinear_tf=True)
Xt_train = tfidf.fit_transform(train.clean_text)
Xt_test = tfidf.transform(test.clean_text)
print(Xb_train.shape, 'nnz/doc', Xb_train.nnz / Xb_train.shape[0])
print(Xt_train.shape, 'nnz/doc', Xt_train.nnz / Xt_train.shape[0])
```

### Model training and held-out evaluation helper
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report

SEED = 42
idx_tr, idx_val = train_test_split(
    np.arange(len(y_all)), test_size=0.1, stratify=y_all, random_state=SEED
)
y_tr, y_val = y_all[idx_tr], y_all[idx_val]
Xt_tr, Xt_val = Xt_all[idx_tr], Xt_all[idx_val]

results = {}
def evaluate(name, model, Xtr, Xval, Xte, features, best_params=None, cv_score=None):
    t0 = time.time()
    model.fit(Xtr, y_tr)
    fit_s = time.time() - t0
    out = {'features': features, 'best_params': best_params,
           'cv_f1_macro': cv_score, 'fit_seconds': round(fit_s, 1)}
    pv = model.predict(Xval)
    out['val_accuracy'] = accuracy_score(y_val, pv)
    t0 = time.time()
    pt = model.predict(Xte)
    out['predict_ms_per_1000'] = round((time.time() - t0) / len(y_test) * 1e6, 1)
    p, r, f, _ = precision_recall_fscore_support(y_test, pt, average='weighted')
    pm, rm, fm, _ = precision_recall_fscore_support(y_test, pt, average='macro')
    out.update(test_accuracy=accuracy_score(y_test, pt), precision=p,
               recall=r, f1=f, f1_macro=fm)
    out['per_class'] = classification_report(
        y_test, pt, target_names=classes, output_dict=True
    )
    out['confusion_matrix'] = confusion_matrix(y_test, pt).tolist()
    results[name] = out
    joblib.dump(model, f'models/{name}.joblib')
    print(f"{name:28s} val_acc={out['val_accuracy']:.4f} "
          f"test_acc={out['test_accuracy']:.4f} F1={f:.4f} fit={fit_s:.0f}s")
    return model
```

### Model selection and probability calibration
```python
import json
import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV

res = json.load(open('reports/results.json'))
cols = ['features', 'test_accuracy', 'precision', 'recall', 'f1',
        'f1_macro', 'val_accuracy', 'fit_seconds', 'predict_ms_per_1000']
df = pd.DataFrame({k: {c: v[c] for c in cols} for k, v in res.items()}).T
for c in cols[1:]:
    df[c] = df[c].astype(float)
df = df.sort_values('f1', ascending=False)

cands = df[df.features.str.startswith('TF-IDF')]
best_f1 = cands.f1.max()
close = cands[cands.f1 >= best_f1 - 0.002]
best = close.sort_values(['predict_ms_per_1000']).index[0]
print('Candidates within 0.002 F1 of the best:', list(close.index))
print('Selected:', best)

model = joblib.load(f'models/{best}.joblib')
needs_cal = not hasattr(model, 'predict_proba')
if needs_cal:
    X = sp.load_npz('models/features/tfidf_train.npz')
    y = np.load('models/features/y_train.npy')
    base = type(model)(**model.get_params())
    model = CalibratedClassifierCV(base, cv=5, method='sigmoid').fit(X, y)
    Xt = sp.load_npz('models/features/tfidf_test.npz')
    yt = np.load('models/features/y_test.npy')
    print('calibrated test accuracy:', (model.predict(Xt) == yt).mean())
joblib.dump(model, 'models/best_model.joblib')
```

## Tables

### Dataset and exploratory statistics
| Statistic | Training | Test |
| --- | --- | --- |
| Dataset shape | (120,000, 3) | (7,600, 3) |
| Articles per class | 30,000 each | not detailed in notebook output |
| Missing values | 0 | 0 |
| Duplicate rows | 0 | 0 |
| Duplicate texts | 0 | 0 |
| Mean word count | 37.84745 | not computed |
| Word-count range | 8 - 177 | not computed |

### Feature matrices from notebook 03
| Representation | Shape | Nonzeros per document |
| --- | --- | --- |
| Bag of Words (max_features=50,000, min_df=3) | (120000, 25393) | 19.49595 |
| TF-IDF (1-2 grams, max_features=100,000, min_df=3) | (120000, 100000) | 28.4295 |

### Seven-configuration accuracy comparison from `PROJECT_REPORT.md`
| Model / Configuration | Test accuracy |
| --- | ---: |
| LinearSVC + TF-IDF | 92.36% |
| Logistic Regression + TF-IDF | 92.00% |
| Multinomial Naive Bayes + TF-IDF | 90.76% |
| Logistic Regression + BoW | 90.24% |
| Naive Bayes + BoW | 90.18% |
| Random Forest | 89.89% |
| Histogram-based Gradient Boosting | 89.04% |

### Full observed metrics from `reports/results.json`
| Model | Features | Validation accuracy | Test accuracy | Precision | Recall | Weighted F1 | Macro F1 | Fit seconds | Prediction ms/1000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear_svc | TF-IDF | 0.9292 | 0.9236 | 0.9235 | 0.9236 | 0.9234 | 0.9234 | 2.4 | 0.2 |
| logistic_regression | TF-IDF | 0.9257 | 0.9200 | 0.9199 | 0.9200 | 0.9199 | 0.9199 | 12.8 | 0.2 |
| naive_bayes | TF-IDF | 0.9175 | 0.9076 | 0.9076 | 0.9076 | 0.9074 | 0.9074 | 0.0 | 0.3 |
| logistic_regression_bow | BoW | 0.9108 | 0.9024 | 0.9023 | 0.9024 | 0.9023 | 0.9023 | 3.2 | 0.2 |
| naive_bayes_bow | BoW | 0.9112 | 0.9018 | 0.9015 | 0.9018 | 0.9016 | 0.9016 | 0.0 | 0.2 |
| random_forest | TF-IDF (chi2 top 20k) | 0.9030 | 0.8989 | 0.8987 | 0.8989 | 0.8986 | 0.8986 | 219.6 | 56.1 |
| gradient_boosting | TF-IDF + SVD(200) | 0.8967 | 0.8904 | 0.8900 | 0.8904 | 0.8901 | 0.8901 | 49.9 | 34.5 |

### Per-class F1 of the selected LinearSVC model
| Business | Sci/Tech | Sports | World |
| ---: | ---: | ---: | ---: |
| 0.8952 | 0.9026 | 0.9707 | 0.9251 |

### Confusion matrix of the selected LinearSVC model
| True / Predicted | Business | Sci/Tech | Sports | World |
| --- | ---: | ---: | ---: | ---: |
| Business | 1692 | 148 | 13 | 47 |
| Sci/Tech | 110 | 1728 | 14 | 48 |
| Sports | 8 | 8 | 1871 | 13 |
| World | 70 | 45 | 57 | 1728 |

### Final calibrated model metadata and sample inference
| Field / sample | Observed value |
| --- | --- |
| Selected model | LinearSVC (calibrated) |
| Features | TF-IDF |
| Best parameters | C = 0.3 |
| Uncalibrated test accuracy | 0.9236 |
| Calibrated test accuracy | 0.9238 |
| F1 | 0.9234 |
| Calibration training size | 120,000 |
| Business example confidence | 0.97 |
| Sports example confidence | 1.00 |
| Sci/Tech example confidence | 1.00 |
| World example confidence | 0.94 |
| End-to-end accuracy on 1,000 raw test articles | 0.928 |

### Team responsibilities from `PROJECT_REPORT.md`
| Member | Primary responsibility |
| --- | --- |
| Kartiken Sharma | Dataset collection and exploratory data analysis |
| Huwaid Siddqui | Text preprocessing and feature extraction |
| Naman Chanana | Model building and model evaluation |
| Mohit Kumar Yadav | Model comparison, selection and prediction system |
| Ishika Garg | Frontend development, integration and documentation |

## Figures

| path | caption |
| --- | --- |
| /Users/kartiken/Downloads/CLNLP-LAB/EXP-9/reports/model_comparison.png | Bar chart comparing test-set accuracy, precision, recall and F1 across the seven AG News model and feature configurations. |
| /Users/kartiken/Downloads/CLNLP-LAB/EXP-9/reports/model_speed.png | Horizontal bar charts of training time in seconds and prediction time in milliseconds per 1,000 documents for all seven models. |
| /Users/kartiken/Downloads/CLNLP-LAB/EXP-9/reports/confusion_matrices.png | Grid of confusion matrices for all seven classifiers with Business, Sci/Tech, Sports and World as axes. |
| /Users/kartiken/Downloads/CLNLP-LAB/EXP-9/reports/top3_confusion.png | Confusion matrices for the three strongest configurations: LinearSVC, logistic regression and Naive Bayes on TF-IDF. |
| /Users/kartiken/Downloads/CLNLP-LAB/report_file/assets/exp9_eda_01.png | Count plot titled "Distribution of News Categories" showing balanced AG News classes. |
| /Users/kartiken/Downloads/CLNLP-LAB/report_file/assets/exp9_eda_02.png | Histogram with KDE titled "Distribution of News Article Length" showing article word counts. |
| /Users/kartiken/Downloads/CLNLP-LAB/report_file/assets/exp9_eda_04.png | Bar chart titled "Top 20 Most Frequent Words" in the AG News training text. |
