# News Classification using NLP — Project Report

**Course:** Natural Language Processing (NLP)  
**Project:** Automated News Article Classification  
**Team Size:** 5 Members  
**Report Date:** 19 November 2026

---

## 1. Introduction

The rapid growth of digital news has made manual categorization of articles inefficient and error-prone. This project focuses on building an end-to-end Natural Language Processing (NLP) pipeline that automatically classifies news articles into four predefined categories: **World, Sports, Business, and Sci/Tech**.

The project covers the complete machine-learning lifecycle: dataset exploration, text preprocessing, feature extraction, model training and evaluation, model comparison and selection, reusable inference, and integration with a user-facing web application.

**Dataset:** AG News dataset via Hugging Face (`sh0416/ag_news`) containing **120,000 training articles** and **7,600 test articles**, balanced across the four target classes.

---

## 2. Team Contribution Breakdown

| Team Member | Primary Responsibility |
| --- | --- |
| **Kartiken Sharma** | Dataset Collection & Exploratory Data Analysis |
| **Huwaid Siddqui** | Text Preprocessing & Feature Extraction |
| **Naman Chanana** | Model Building & Model Evaluation |
| **Mohit Kumar Yadav** | Model Comparison, Selection & Prediction System |
| **Ishika Garg** | Frontend Development, Integration & Documentation |

---

## 3. Detailed Contributions

### 3.1 Kartiken Sharma — Dataset Collection & Exploratory Data Analysis

**Primary responsibilities:**

- Loaded the AG News dataset using `datasets.load_dataset("sh0416/ag_news")`.
- Converted the Hugging Face dataset objects into pandas DataFrames for analysis.
- Performed structural analysis including dataset shape, data types, column profiling, and descriptive statistics.
- Mapped numeric labels to the four news categories:
  - 1 → World
  - 2 → Sports
  - 3 → Business
  - 4 → Sci/Tech
- Combined article titles and descriptions into a unified text representation.
- Performed data-quality checks including missing-value detection, duplicate-row detection, and duplicate-text analysis.
- Conducted exploratory statistical and visual analysis:
  - category distribution
  - word-count distribution
  - article length by category
  - frequently occurring words

**Tools/Libraries:** pandas, NumPy, Matplotlib, Seaborn, `collections.Counter`, `re`, Hugging Face Datasets

**Key file:** `01_EDA.ipynb`

**Outcome:** Established the analytical understanding of the AG News dataset and verified its structure, quality, and class distribution before preprocessing and modelling.

---

### 3.2 Huwaid Siddqui — Text Preprocessing & Feature Extraction

**Primary responsibilities:**

- Designed and implemented the text-cleaning pipeline:
  - lowercasing
  - HTML-tag removal
  - URL removal
  - special-character cleaning
  - tokenization
  - stopword removal
  - Porter stemming
  - handling of empty or malformed text
- Applied a consistent preprocessing pipeline to both training and test data.
- Implemented the feature representations used by the classical machine-learning models:
  - Bag of Words (BoW)
  - TF-IDF with unigram and bigram features
- Prepared the fitted feature-extraction artifacts for reuse during model training and inference.
- Helped ensure that the preprocessing used during inference remained consistent with the preprocessing used during training.

**Tools/Libraries:** NLTK, `re`, scikit-learn `CountVectorizer`, scikit-learn `TfidfVectorizer`

**Key files:**

- `02_Text Preprocessing.ipynb`
- `03_Feature_Extraction.ipynb`
- `src/preprocess.py`

**Outcome:** Produced clean and consistent numerical representations of the news text suitable for classical machine-learning classification.

---

### 3.3 Naman Chanana — Model Building & Model Evaluation

**Primary responsibilities:**

- Built and evaluated multiple machine-learning classifiers using the extracted text features.
- Evaluated configurations involving:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Linear Support Vector Classification (LinearSVC)
  - Random Forest
  - Histogram-based Gradient Boosting
- Worked with both Bag-of-Words and TF-IDF feature representations where applicable.
- Performed stratified train/validation evaluation to preserve class balance.
- Evaluated model performance using:
  - accuracy
  - precision
  - recall
  - F1-score
  - confusion matrices
  - classification reports
- Analyzed model behaviour and class-wise prediction performance.
- Compared the effectiveness of different model and feature combinations.

**Tools/Libraries:** scikit-learn, NumPy, pandas, Matplotlib, Seaborn

**Key file:** `04_Model_Training_Evaluation.ipynb`

**Outcome:** Established the comparative performance of the candidate classifiers and provided the quantitative evidence required for final model selection.

---

### 3.4 Mohit Kumar Yadav — Model Comparison, Selection & Prediction System

#### Model Comparison & Selection

**Primary responsibilities:**

- Consolidated model results into a unified comparison of classification performance.
- Compared candidate configurations using accuracy, weighted F1-score, inference behaviour, and practical model characteristics.
- Visualized model performance using comparison plots and confusion matrices.
- Selected **LinearSVC with TF-IDF features** as the strongest-performing configuration.
- Considered the trade-off between predictive performance, inference speed, and model complexity.

#### Prediction System

- Built the reusable end-to-end inference flow:

```text
Raw News Text
      ↓
Shared Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Selected LinearSVC Model
      ↓
Probability Calibration
      ↓
Predicted News Category
```

- Reused the same preprocessing logic used during model development.
- Serialized and integrated the trained model and feature-extraction artifacts.
- Developed reusable prediction functionality through `predict(text)`.
- Integrated confidence/probability information into the inference output.

**Tools/Libraries:** scikit-learn, joblib, NumPy

**Key files:**

- `05_Model_Comparison_Prediction.ipynb`
- `src/predict.py`
- `models/`

**Outcome:** Produced the final deployable prediction pipeline using the best-performing model configuration.

---

### 3.5 Ishika Garg — Frontend Development, Integration & Documentation

#### Frontend Development

**Primary responsibilities:**

- Developed the user-facing web interface for the news classifier.
- Enabled users to:
  - enter or paste news text
  - submit it for classification
  - view the predicted category
  - inspect confidence/probability information and model details
- Organized the interface to provide a simple and understandable classification workflow.

#### Application Integration

- Integrated the reusable prediction pipeline with the Flask backend.
- Connected frontend requests to the trained model through the application's prediction endpoints.
- Supported local/demo execution of the complete classification system.
- Helped maintain dependency and setup information required to reproduce the application environment.

#### Documentation

- Organized project documentation and submission material.
- Documented setup, installation, usage, architecture, and project results.
- Helped organize the repository into a submission-ready structure.

**Tools/Libraries:** Flask, HTML, CSS, JavaScript, joblib

**Key directory:** `app/`

**Outcome:** Delivered the user-facing layer of the classifier and helped package the technical work into a complete demonstrable project.

---

## 4. System Architecture

```text
                         AG News Dataset
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ Dataset Analysis & EDA  │
                  │    Kartiken Sharma      │
                  └────────────┬────────────┘
                               │
                               ▼
                ┌─────────────────────────────┐
                │ Preprocessing & Features    │
                │      Huwaid Siddqui         │
                │ Porter Stem + BoW / TF-IDF  │
                └──────────────┬──────────────┘
                               │
                               ▼
                ┌─────────────────────────────┐
                │ Model Training & Evaluation │
                │       Naman Chanana         │
                │  NB · LR · SVM · RF · HGB  │
                └──────────────┬──────────────┘
                               │
                               ▼
                ┌─────────────────────────────┐
                │ Model Selection & Inference │
                │    Mohit Kumar Yadav        │
                │ LinearSVC + TF-IDF          │
                └──────────────┬──────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ Flask Web Application   │
                  │      Ishika Garg        │
                  └────────────┬────────────┘
                               │
                               ▼
                  Predicted News Category
```

---

## 5. Project Files

| File / Directory | Description | Primary Responsibility |
| --- | --- | --- |
| `01_EDA.ipynb` | Dataset exploration and analysis | Kartiken Sharma |
| `02_Text Preprocessing.ipynb` | Initial text-cleaning experiments | Huwaid Siddqui |
| `03_Feature_Extraction.ipynb` | Final preprocessing, BoW and TF-IDF feature extraction | Huwaid Siddqui |
| `04_Model_Training_Evaluation.ipynb` | Model training and evaluation | Naman Chanana |
| `05_Model_Comparison_Prediction.ipynb` | Model comparison, final selection and prediction experiments | Mohit Kumar Yadav |
| `src/preprocess.py` | Shared preprocessing implementation | Shared pipeline |
| `src/predict.py` | Reusable inference pipeline | Mohit Kumar Yadav |
| `app/` | Flask backend and frontend application | Ishika Garg |
| `models/` | Serialized model and feature artifacts | Model/inference pipeline |
| `reports/` | Evaluation results, plots and confusion matrices | Team evaluation output |
| `MODEL_CARD.md` | Final model, performance, compatibility and limitations | Project documentation |
| `README.md` | Repository setup, usage and results overview | Team documentation |
| `PROJECT_REPORT.md` | Complete project report | Team documentation |

---

## 6. Model Comparison and Final Results

Seven model/feature configurations were evaluated during the project.

| Model / Configuration | Test Accuracy |
| --- | ---: |
| **LinearSVC + TF-IDF** | **92.36%** |
| Logistic Regression + TF-IDF | 92.00% |
| Multinomial Naive Bayes + TF-IDF | 90.76% |
| Logistic Regression + BoW | 90.24% |
| Naive Bayes + BoW | 90.18% |
| Random Forest | 89.89% |
| Histogram-based Gradient Boosting | 89.04% |

The final selected model is a **calibrated LinearSVC using TF-IDF unigram and bigram features**.

Final evaluation on the **7,600-article AG News test set**:

- **Accuracy:** 92.36%
- **Weighted F1-score:** 0.923

The selected model provided the strongest overall classification performance among the evaluated configurations while remaining suitable for fast inference in the web application.

---

## 7. Reproducibility and Model Compatibility

The deployed model and preprocessing artifacts are stored in the `models/` directory and loaded by the reusable inference pipeline.

The serialized scikit-learn artifacts were created using:

```text
scikit-learn 1.9.1
```

The corresponding version is pinned in `requirements.txt` to improve compatibility when loading the distributed model artifacts.

The final preprocessing pipeline uses **Porter stemming** and TF-IDF feature extraction. Word2Vec or contextual transformer embeddings are not part of the deployed classification pipeline.

Additional model details, intended use, and limitations are documented in [`MODEL_CARD.md`](MODEL_CARD.md).

---

## 8. Conclusion

The project demonstrates a complete classical NLP workflow for automated news classification, progressing from raw text data to a reusable and user-facing prediction system.

The five-member team contributed across complementary stages of the pipeline:

- **Kartiken Sharma** established the dataset and exploratory-analysis foundation.
- **Huwaid Siddqui** developed the preprocessing and feature-extraction pipeline.
- **Naman Chanana** trained and evaluated the candidate machine-learning models.
- **Mohit Kumar Yadav** compared the models, selected the final configuration, and built the reusable prediction pipeline.
- **Ishika Garg** integrated the classifier into the web application and organized the final documentation and presentation layer.

The final system uses a calibrated **LinearSVC with TF-IDF features**, achieving **92.36% accuracy** and a **0.923 weighted F1-score** on the 7,600-article AG News test set.

The completed repository therefore demonstrates the full lifecycle of a practical NLP application:

```text
Data
  ↓
EDA
  ↓
Preprocessing
  ↓
Feature Engineering
  ↓
Model Training
  ↓
Evaluation & Selection
  ↓
Reusable Inference
  ↓
Web Application
```

---

*Prepared as part of the Natural Language Processing team project.*
