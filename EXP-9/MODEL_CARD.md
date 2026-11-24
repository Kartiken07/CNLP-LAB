# Model Card

## Model Overview

The deployed classifier is a **Linear Support Vector Classifier (LinearSVC)** trained on TF-IDF features for four-class AG News topic classification.

### Classes

- World
- Sports
- Business
- Sci/Tech

## Performance

The selected model achieved:

| Metric | Result |
| --- | ---: |
| Test Accuracy | **92.36%** |
| Training Samples | 120,000 |
| Test Samples | 7,600 |

Among the evaluated configurations, LinearSVC with TF-IDF achieved the highest test accuracy.

## Preprocessing

The shared preprocessing pipeline includes text normalization and **Porter stemming** before feature extraction.

The deployed model uses the same preprocessing implementation exposed by the project inference pipeline.

## Model Artifacts

Serialized model and preprocessing artifacts are stored in:

```text
models/
```

These artifacts are loaded by the reusable inference code in:

```text
src/predict.py
```

## Compatibility

The distributed scikit-learn artifacts were created with:

```text
scikit-learn 1.9.1
```

For reliable artifact loading, the repository pins this version in `requirements.txt`.

Serialized scikit-learn estimators are not intended to provide guaranteed compatibility across arbitrary scikit-learn versions.

## Intended Use

The model is intended for educational and experimental classification of English news text into the four AG News categories.

It is not designed for high-stakes decision making or classification outside the domain represented by the training dataset.

## Limitations

- The classifier is trained specifically on AG News categories.
- Predictions outside the news domain may be unreliable.
- TF-IDF features do not model contextual language semantics in the same way as modern contextual language models.
- Reported performance reflects the project's AG News test split and should not be interpreted as universal classification performance.
