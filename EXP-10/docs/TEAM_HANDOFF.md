# Low-Resource NMT (English → Assamese): Team Handoff

## Status

The English-to-Assamese EXP 10 implementation, GPU training, and quantitative evaluation are complete.

The model was fine-tuned and evaluated by **Naman Chanana** on Kaggle using an NVIDIA Tesla T4 GPU.

Repository:

`Kartiken07/LOW-RESOURCE-NEURAL-MACHINE-TRANSLATION`

### Current Project State

| Item | State |
| --- | --- |
| `src/preprocess.py` | Complete — preprocessing executed successfully on the Samanantar corpus |
| `src/train.py` | Complete — full 2-epoch GPU training run completed |
| `src/evaluate.py` | Complete — pretrained and fine-tuned models evaluated on the same held-out examples |
| `notebooks/exp10_nmt_kaggle.ipynb` | Available for the full interactive Kaggle workflow |
| `notebooks/run_all_kaggle.ipynb` | Available for the one-click Kaggle pipeline |
| `notebooks/show_results.ipynb` | Available for displaying saved experiment results |
| `results/metrics.csv` | Contains real BLEU and chrF results |
| `results/sample_translations.csv` | Contains real pretrained and fine-tuned translations |
| `results/training_log.csv` | Contains the real training/evaluation history |
| `results/training_time_minutes.txt` | Contains measured training duration |
| `results/experiment_summary.md` | Contains the complete experiment summary and error analysis |
| Dataset | Not stored in Git; downloaded automatically from Hugging Face |
| Trained checkpoint | Not stored in Git because of its size (~2.4 GB) |

The WMT 2026 data originally suggested in the project guide was not used because its test data requires participant registration. The experiment instead uses the openly available AI4Bharat Samanantar English–Assamese corpus.

---

## Assignment

| Task | Owner | Status |
| --- | --- | --- |
| Train and evaluate the model on Kaggle | **Naman Chanana** | **Completed** |
| Record and contribute real experimental results | **Naman Chanana** | **Completed** |
| Update README with settings, metrics and qualitative analysis | **Naman Chanana** | **Completed** |
| Short project report | Mohit / team | Pending |
| Classroom presentation and live demo | Mohit / team | Pending |

The project guide requires one pretrained model, one language pair, and a before/after fine-tuning comparison. This experiment follows that requirement using **NLLB-200 distilled 600M** for **English → Assamese**.

---

# Completed Experiment

## Platform

The full run was performed on:

```text
Platform: Kaggle
GPU: NVIDIA Tesla T4
```

GPU availability was confirmed by the training pipeline before fine-tuning.

---

## Dataset

Dataset:

**AI4Bharat Samanantar — Assamese configuration**

The raw dataset contained:

```text
141,227 English–Assamese pairs
```

Preprocessing produced:

| Stage | Pairs |
| --- | ---: |
| Raw | 141,227 |
| After empty removal | 141,227 |
| After length filtering | 141,206 |
| After length-ratio filtering | 139,713 |
| After deduplication | 139,707 |
| Random subsample | **40,000** |

The selected data was shuffled using seed `42` and split 80:10:10:

```text
Training      32,000
Validation     4,000
Test           4,000
```

Dataset files are intentionally excluded from Git.

---

## Training Configuration

The experiment used the planned configuration without reducing the dataset, epochs, batch size, or sequence length.

| Setting | Actual Value |
| --- | --- |
| Base model | `facebook/nllb-200-distilled-600M` |
| Direction | English → Assamese |
| Source language | `eng_Latn` |
| Target language | `asm_Beng` |
| Epochs | 2 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Effective batch size | 16 |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Maximum sequence length | 128 |
| Seed | 42 |
| Mixed precision | FP16 |
| Validation | After every epoch |
| Best-model criterion | Validation loss |
| GPU | NVIDIA Tesla T4 |
| Training time | **54.3 minutes** |

Training completed all:

```text
4,000 optimization steps
```

The final recorded validation loss was approximately:

```text
1.9105
```

---

# Evaluation Results

Both models were evaluated on the **same 1,000 held-out test pairs**.

Metrics were calculated using SacreBLEU.

| Model | BLEU | chrF |
| --- | ---: | ---: |
| Pretrained NLLB | 7.45 | 31.82 |
| **Fine-tuned NLLB** | **9.80** | **35.29** |

### Improvement

```text
BLEU:  7.45  → 9.80   (+2.35)
chrF: 31.82 → 35.29  (+3.47)
```

Relative improvement:

- BLEU: approximately **31.5%**
- chrF: approximately **10.9%**

The increase in both metrics demonstrates measurable adaptation to the English–Assamese training corpus.

---

# Qualitative Results

The repository stores five example translations from each model in:

```text
results/sample_translations.csv
```

Several examples show meaningful improvement after fine-tuning.

### Example: Public servants

English:

> We are public servants.

Reference:

> আমি চৰকাৰৰ দ্বাৰা নিযুক্ত কৰ্মচৰী।

Pretrained:

> আমি ৰাজহুৱা সেৱক।

Fine-tuned:

> আমি ৰাজহুৱা কৰ্মচাৰী।

The fine-tuned output uses terminology closer to the reference.

---

### Example: Session logout

English:

> Log out %s of this session to log in as a different user

Both models preserve the `%s` placeholder. The fine-tuned output also moves closer to the reference wording.

---

### Example: Start Menu

English:

> In my &Start Menu Programs folder

Pretrained:

> মোৰ &Start Menu Programs ফোল্ডাৰত

Fine-tuned:

> মোৰ &Start মেনু প্ৰোগ্ৰাম ফোল্ডাৰত

The fine-tuned model translates more of the English interface phrase into Assamese.

---

# Error Analysis

The fine-tuned model improved overall BLEU and chrF, but not every sentence improved.

One evaluation example exposed a clear hallucination.

English:

> Rather than say what might be discouraging, we could strive to be encouraging.

Reference:

> নিৰন্তৰে ইজনে সিজনক সহায় কৰক

The pretrained model generated an unrelated editing marker, while the fine-tuned model generated unrelated religious content.

This is important to report rather than hide: the experiment demonstrates **measurable aggregate improvement, not production-level translation reliability**.

Likely contributing factors include:

- low-resource training conditions
- noisy or loosely aligned sentence pairs
- limited 40,000-pair training subset
- domain mismatch
- only two fine-tuning epochs

See `results/experiment_summary.md` for the complete result analysis.

---

# Reproducibility Artifacts

The following lightweight experiment outputs are committed:

```text
results/
├── metrics.csv
├── sample_translations.csv
├── training_log.csv
├── training_time_minutes.txt
└── experiment_summary.md
```

The following are deliberately excluded:

```text
data/
nllb-en-as/
*.safetensors
*.bin
*.pt
```

The trained checkpoint is approximately 2.4 GB and should remain on Kaggle or another external model-hosting service rather than GitHub.

---

# Reproducing the Experiment

## Kaggle

Recommended workflow:

1. Import `notebooks/run_all_kaggle.ipynb` or `notebooks/exp10_nmt_kaggle.ipynb`.
2. Enable a GPU accelerator.
3. Enable Internet access.
4. Run preprocessing.
5. Evaluate the pretrained NLLB baseline.
6. Fine-tune for two epochs.
7. Evaluate the fine-tuned model on the same 1,000 test examples.
8. Inspect the generated translations.

Actual full-run configuration:

```text
MAX_ROWS = 40000
N_EVAL = 1000
epochs = 2
batch size = 4
gradient accumulation = 4
learning rate = 2e-5
max length = 128
seed = 42
```

---

# Notes From the Actual Kaggle Run

The experiment completed successfully on a Tesla T4.

Non-fatal warnings observed during execution included:

- Hugging Face Hub requests were unauthenticated.
- Transformers reported that `max_new_tokens=128` takes precedence over the model's `max_length=200`.
- A pandas future warning was emitted while updating the result CSV.
- Kaggle's preinstalled environment reported unrelated package-version conflicts after dependency installation.

None of these warnings prevented preprocessing, training, model saving, or evaluation from completing.

The original imported one-click notebook also encountered a Kaggle notebook-metadata/kernel issue before training began. The experiment was subsequently run successfully in a fresh Kaggle notebook using the repository pipeline commands.

---

# Remaining Team Work

The core ML experiment is complete.

Remaining coursework tasks are primarily presentation/reporting work:

- [ ] Prepare the short project report
- [ ] Prepare the classroom presentation
- [ ] Prepare the live English → Assamese demo
- [ ] Add any screenshots/plots required by the course submission

The report should use the real experiment values recorded here and in `results/`.

Do **not** estimate or replace the measured metrics.

---

# Completed Checklist

- [x] Kaggle GPU run completed
- [x] NVIDIA Tesla T4 confirmed
- [x] 40,000-pair preprocessing completed
- [x] 32,000 training pairs used
- [x] Two training epochs completed
- [x] Pretrained baseline evaluated
- [x] Fine-tuned model evaluated
- [x] Both models evaluated on the same 1,000 test examples
- [x] Real BLEU and chrF recorded
- [x] Training time recorded
- [x] Training log preserved
- [x] Sample translations preserved
- [x] README updated with real results
- [x] Error analysis documented
- [x] Dataset excluded from Git
- [x] Trained model/checkpoints excluded from Git
- [x] No credentials or access tokens required by the experiment

---

# Results for Report / Presentation

Use these exact measured values:

```text
Model:
facebook/nllb-200-distilled-600M

Language pair:
English → Assamese

Dataset:
AI4Bharat Samanantar

Selected pairs:
40,000

Train / validation / test:
32,000 / 4,000 / 4,000

Evaluation pairs:
1,000

Epochs:
2

GPU:
NVIDIA Tesla T4

Training time:
54.3 minutes

Pretrained:
BLEU = 7.45
chrF = 31.82

Fine-tuned:
BLEU = 9.80
chrF = 35.29

Improvement:
BLEU +2.35
chrF +3.47

Final validation loss:
~1.9105
```

These values come from the actual Kaggle experiment and should be used consistently in the README, report and presentation.