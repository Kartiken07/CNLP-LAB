# Low-Resource Neural Machine Translation: English → Assamese

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-NLLB--200-FFD21E?logo=huggingface&logoColor=black" alt="NLLB-200">
  <img src="https://img.shields.io/badge/English%20%E2%86%92%20Assamese-NMT-6A5ACD" alt="English to Assamese">
  <img src="https://img.shields.io/badge/Kaggle-Tesla%20T4-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle Tesla T4">
  <img src="https://img.shields.io/badge/BLEU-9.80-2EA44F" alt="BLEU 9.80">
  <img src="https://img.shields.io/badge/chrF-35.29-2EA44F" alt="chrF 35.29">
  <a href="https://github.com/Kartiken07/LOW-RESOURCE-NEURAL-MACHINE-TRANSLATION/actions/workflows/ci.yml">
    <img src="https://github.com/Kartiken07/LOW-RESOURCE-NEURAL-MACHINE-TRANSLATION/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://www.kaggle.com/code/namanchanana/nllb-200-english-to-assamese-fine-tuning-samanan">
    <img src="https://img.shields.io/badge/Kaggle-Training%20Run-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle Training Run">
  </a>

</p>

Fine-tuning **NLLB-200 distilled 600M** for low-resource English-to-Assamese (`en-as`) neural machine translation using the AI4Bharat Samanantar parallel corpus.

This repository contains the complete EXP 10 pipeline:

**Data preparation → Pretrained baseline → Fine-tuning → Evaluation → Qualitative analysis**

## Contents

- [Results](#results)
- [Experiment Configuration](#experiment-configuration)
- [Dataset](#dataset)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Qualitative Examples](#qualitative-examples)
- [Error Analysis](#error-analysis)
- [Results Artifacts](#results-artifacts)
- [How to Reproduce](#how-to-reproduce)
- [Testing](#testing)
- [Repository Structure](#repository-structure)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References and Attribution](#references-and-attribution)


## Results

The experiment was successfully trained and evaluated on a **Kaggle NVIDIA Tesla T4 GPU**.

Both models were evaluated on the **same 1,000 held-out test pairs**.

| Model | BLEU | chrF |
| --- | ---: | ---: |
| Pretrained NLLB-200 distilled 600M | 7.45 | 31.82 |
| **Fine-tuned NLLB** | **9.80** | **35.29** |

### Improvement

- **BLEU:** 7.45 → 9.80 (**+2.35 points, ~31.5% relative improvement**)
- **chrF:** 31.82 → 35.29 (**+3.47 points, ~10.9% relative improvement**)

Fine-tuning therefore produced a measurable improvement on both evaluation metrics.

> The model remains an experimental low-resource translation system. Metric improvements do not imply that every generated translation is correct; qualitative evaluation identified both improved outputs and occasional hallucination.

---

## Experiment Configuration

| Setting | Value |
| --- | --- |
| Base model | `facebook/nllb-200-distilled-600M` |
| Translation direction | English → Assamese |
| Source language code | `eng_Latn` |
| Target language code | `asm_Beng` |
| Dataset | AI4Bharat Samanantar |
| Selected pairs | 40,000 |
| Training pairs | 32,000 |
| Validation pairs | 4,000 |
| Test pairs | 4,000 |
| Evaluation pairs | 1,000 |
| Epochs | 2 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Effective batch size | 16 |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Maximum sequence length | 128 |
| Seed | 42 |
| Precision | FP16 |
| GPU | NVIDIA Tesla T4 |
| Training time | **54.3 minutes** |
| Final validation loss | **1.9105** |

The best model is selected using validation loss.

---

## Dataset

The experiment uses the Assamese configuration of the open **AI4Bharat Samanantar** parallel corpus.

The raw dataset contained **141,227 English–Assamese sentence pairs**.

### Preprocessing

`src/preprocess.py` performs:

1. Unicode NFC normalization
2. Control/invisible-character removal
3. Whitespace normalization
4. Empty translation removal
5. Length filtering
6. Extreme source/target length-ratio filtering
7. Duplicate-pair removal
8. Deterministic random subsampling
9. 80:10:10 train/validation/test splitting

The actual run produced:

| Stage | Pairs |
| --- | ---: |
| Raw | 141,227 |
| After empty removal | 141,227 |
| After length filtering | 141,206 |
| After length-ratio filtering | 139,713 |
| After deduplication | 139,707 |
| Selected for experiment | **40,000** |

Final split:

```text
Training      32,000
Validation     4,000
Test           4,000
```

The split and subsampling use seed `42`.

Dataset files are intentionally **not stored in this repository** and are downloaded automatically during preprocessing.

### Licences

- **Samanantar:** CC BY-NC 4.0
- **NLLB-200:** CC-BY-NC-4.0

This repository is intended for academic/research use.

---

## Model

The project fine-tunes:

```text
facebook/nllb-200-distilled-600M
```

NLLB-200 is a multilingual sequence-to-sequence translation model.

For this experiment:

```text
English   → eng_Latn
Assamese  → asm_Beng
```

Tokenization and generation use the corresponding NLLB language codes.

---

## Training

Training is implemented in:

```text
src/train.py
```

The full experiment used:

```bash
python src/train.py \
  --data-dir data/processed \
  --output-dir nllb-en-as \
  --epochs 2 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 2e-5 \
  --max-length 128 \
  --seed 42
```

Training uses:

- `Seq2SeqTrainer`
- AdamW-based optimization
- gradient accumulation
- FP16 on CUDA
- validation after each epoch
- checkpointing after each epoch
- best-model loading by validation loss
- deterministic seed configuration

Training completed **4,000 optimization steps** in **54.3 minutes** on an NVIDIA Tesla T4.

The final validation loss was approximately:

```text
1.9105
```

---

## Evaluation

Evaluation is implemented in:

```text
src/evaluate.py
```

Both the pretrained and fine-tuned models were evaluated on exactly the same first **1,000 examples** from the held-out test split.

Metrics are calculated with **SacreBLEU**:

- BLEU
- chrF

Baseline:

```bash
python src/evaluate.py \
  --model facebook/nllb-200-distilled-600M \
  --label "Pretrained (before fine-tuning)" \
  --max-samples 1000
```

Fine-tuned model:

```bash
python src/evaluate.py \
  --model nllb-en-as \
  --label "Fine-tuned" \
  --max-samples 1000
```

---

## Qualitative Examples

### Example 1

**English**

> We are public servants.

**Reference**

> আমি চৰকাৰৰ দ্বাৰা নিযুক্ত কৰ্মচৰী।

**Pretrained**

> আমি ৰাজহুৱা সেৱক।

**Fine-tuned**

> আমি ৰাজহুৱা কৰ্মচাৰী।

The fine-tuned model uses terminology closer to the reference.

### Example 2

**English**

> Log out %s of this session to log in as a different user

**Pretrained**

> অন্য ব্যৱহাৰকৰ্তাৰূপে লগ ইন কৰিবলৈ এই অধিৱেশনৰ %s ৰ লগ আউট কৰক

**Fine-tuned**

> এটা ভিন্ন ব্যৱহাৰকৰ্তাৰূপে প্ৰৱেশ কৰিবলৈ এই অধিবেশনৰ %s প্ৰস্থান কৰক

Both models preserve the `%s` placeholder, while the fine-tuned output moves closer to the reference wording.

### Example 3

**English**

> In my &Start Menu Programs folder

**Pretrained**

> মোৰ &Start Menu Programs ফোল্ডাৰত

**Fine-tuned**

> মোৰ &Start মেনু প্ৰোগ্ৰাম ফোল্ডাৰত

The fine-tuned model translates more of the interface phrase into Assamese.

### Example 4

**English**

> Expected , or ) after variable name in variable reference but found %1$S.

The fine-tuned model preserves `%1$S` and produces a translation structurally closer to the Assamese reference.

---

## Error Analysis

Fine-tuning improved the aggregate scores, but not every translation improved.

One example clearly demonstrates hallucination.

**English**

> Rather than say what might be discouraging, we could strive to be encouraging.

**Reference**

> নিৰন্তৰে ইজনে সিজনক সহায় কৰক

The pretrained model returned an unrelated editing marker:

> [সম্পাদনা কৰক]

The fine-tuned model instead generated unrelated religious content.

This highlights an important limitation: **higher BLEU and chrF do not guarantee reliable sentence-level translation**.

Potential causes include:

- limited fine-tuning data
- noisy or loosely aligned source/reference pairs
- domain mismatch
- low-resource language coverage
- short fine-tuning duration

The model should therefore be considered an **experimental NMT system**, not a production translator.

---

## Results Artifacts

Real outputs from the Kaggle GPU run are stored under:

```text
results/
├── metrics.csv
├── sample_translations.csv
├── training_log.csv
├── training_time_minutes.txt
└── experiment_summary.md
```

### `metrics.csv`

Contains BLEU and chrF for the pretrained and fine-tuned models.

### `sample_translations.csv`

Contains English source sentences, Assamese references, and generated translations from both models.

### `training_log.csv`

Contains recorded training loss, gradient norms, learning rates, validation metrics, runtime statistics and final training statistics.

### `training_time_minutes.txt`

Contains the measured training duration:

```text
54.3
```

### `experiment_summary.md`

Provides the complete experimental configuration, results, qualitative analysis and limitations.

---

## How to Reproduce

### Kaggle

The easiest method is:

```text
notebooks/run_all_kaggle.ipynb
```

1. Import the notebook into Kaggle.
2. Enable a GPU accelerator.
3. Enable Internet.
4. Run the pipeline.

The notebook performs preprocessing, baseline evaluation, fine-tuning and final evaluation.

A more detailed notebook is also available:

```text
notebooks/exp10_nmt_kaggle.ipynb
```

### Local Setup

GPU execution is strongly recommended.

```bash
git clone https://github.com/Kartiken07/LOW-RESOURCE-NEURAL-MACHINE-TRANSLATION.git
cd LOW-RESOURCE-NEURAL-MACHINE-TRANSLATION

pip install -r requirements.txt
```

Preprocess:

```bash
python src/preprocess.py \
  --hf-dataset samanantar \
  --max-rows 40000 \
  --out-dir data/processed
```

Evaluate pretrained baseline:

```bash
python src/evaluate.py \
  --model facebook/nllb-200-distilled-600M \
  --label "Pretrained (before fine-tuning)" \
  --max-samples 1000
```

Fine-tune:

```bash
python src/train.py \
  --data-dir data/processed \
  --output-dir nllb-en-as \
  --epochs 2 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 2e-5 \
  --max-length 128 \
  --seed 42
```

Evaluate:

```bash
python src/evaluate.py \
  --model nllb-en-as \
  --label "Fine-tuned" \
  --max-samples 1000
```

---

## Testing

The repository includes lightweight automated tests for preprocessing and reproducibility safeguards.

```bash
python -m unittest discover -s tests -v
```

The test suite covers:

- text cleaning and normalization
- empty/non-string handling
- duplicate removal
- deterministic dataset splitting
- expected 80:10:10 split sizes
- notebook format validation
- Python kernel metadata
- exclusion of large training/model artifacts

GitHub Actions runs these checks automatically on pushes and pull requests to `main`.

## Repository Structure

```text
LOW-RESOURCE-NEURAL-MACHINE-TRANSLATION/
│
├── docs/
│   └── TEAM_HANDOFF.md
│
├── notebooks/
│   ├── exp10_nmt_kaggle.ipynb
│   ├── run_all_kaggle.ipynb
│   └── show_results.ipynb
│
├── results/
│   ├── metrics.csv
│   ├── sample_translations.csv
│   ├── training_log.csv
│   ├── training_time_minutes.txt
│   └── experiment_summary.md
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Large Artifacts

The following are intentionally **not committed**:

```text
data/
nllb-en-as/
*.safetensors
*.bin
*.pt
```

The fine-tuned checkpoint is approximately **2.4 GB** and should remain in Kaggle or external model storage rather than Git.

The dataset is downloaded automatically and is also excluded from version control.

---

## Limitations

- Only 40,000 sentence pairs were selected from the available corpus.
- Training was limited to two epochs.
- Quantitative evaluation used 1,000 held-out test pairs.
- Some parallel examples appear noisy or loosely aligned.
- Qualitative evaluation shows occasional hallucination.
- Only one pretrained architecture and one translation direction were evaluated.
- Results should not be interpreted as production-level Assamese translation quality.

---

## Future Work

Possible extensions include:

- training on more of the available Samanantar corpus
- stronger parallel-data filtering
- evaluation on the complete held-out test set
- additional human evaluation by Assamese speakers
- longer or tuned fine-tuning schedules
- domain-specific evaluation
- systematic hallucination analysis

Any future experiment should retain the same held-out evaluation protocol when comparing against the results reported here.

---

## References and Attribution

Model, dataset, evaluation-tool, and framework references are documented in [`REFERENCES.md`](REFERENCES.md).

Third-party resources retain their respective licences and usage terms.

> **Repository licence:** A project-level licence has not yet been specified by the repository owner.

For contribution guidelines, see [`CONTRIBUTING.md`](CONTRIBUTING.md).


## Experiment Provenance

The reported results are from a real Kaggle GPU run and are not estimated.

**Platform:** Kaggle
**GPU:** NVIDIA Tesla T4
**Training time:** 54.3 minutes
**Seed:** 42
**Evaluation:** 1,000 held-out test pairs
**Base model:** NLLB-200 distilled 600M

The raw model checkpoint and training dataset are intentionally excluded from Git.