# English → Assamese NLLB Fine-Tuning Results

## Experiment

This experiment fine-tuned `facebook/nllb-200-distilled-600M` for low-resource English-to-Assamese neural machine translation using the AI4Bharat Samanantar corpus.

The experiment was executed on a Kaggle **Tesla T4 GPU**.

## Dataset

The Samanantar Assamese configuration initially contained 141,227 sentence pairs.

After cleaning and filtering:

| Stage | Pairs |
| --- | ---: |
| Raw | 141,227 |
| Empty removal | 141,227 |
| Length filter | 141,206 |
| Length-ratio filter | 139,713 |
| Deduplication | 139,707 |
| Random subsample | 40,000 |

The 40,000 selected pairs were shuffled with seed `42` and split 80:10:10:

| Split | Pairs |
| --- | ---: |
| Training | 32,000 |
| Validation | 4,000 |
| Test | 4,000 |

Evaluation used the same first **1,000 held-out test pairs** for both the pretrained and fine-tuned models.

## Training Configuration

| Setting | Value |
| --- | --- |
| Base model | `facebook/nllb-200-distilled-600M` |
| Direction | English → Assamese (`en-as`) |
| Source language | `eng_Latn` |
| Target language | `asm_Beng` |
| Epochs | 2 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Effective batch size | 16 |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Maximum sequence length | 128 |
| Random seed | 42 |
| Mixed precision | FP16 |
| Evaluation strategy | Every epoch |
| Best-model criterion | Validation loss |
| GPU | NVIDIA Tesla T4 |
| Training time | 54.3 minutes |

Training completed all **4,000 optimization steps**.

The final recorded validation loss was approximately **1.9105**.

## Evaluation Results

BLEU and chrF were calculated with SacreBLEU.

| Model | Test pairs | BLEU | chrF |
| --- | ---: | ---: | ---: |
| Pretrained NLLB | 1,000 | 7.45 | 31.82 |
| Fine-tuned NLLB | 1,000 | **9.80** | **35.29** |

### Improvement after fine-tuning

- BLEU: **+2.35 points** (~31.5% relative improvement)
- chrF: **+3.47 points** (~10.9% relative improvement)

The increase in both metrics shows that fine-tuning improved translation quality on the held-out English–Assamese evaluation subset.

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

**Reference**

> বেলেগ ব্যৱহাৰকৰ্তা হিচাপে প্ৰৱেশ কৰিবলৈ %s ক বৰ্তমান অধিবেছনৰ পৰা প্ৰস্থান কৰা হ'ব ।

**Pretrained**

> অন্য ব্যৱহাৰকৰ্তাৰূপে লগ ইন কৰিবলৈ এই অধিৱেশনৰ %s ৰ লগ আউট কৰক

**Fine-tuned**

> এটা ভিন্ন ব্যৱহাৰকৰ্তাৰূপে প্ৰৱেশ কৰিবলৈ এই অধিবেশনৰ %s প্ৰস্থান কৰক

Both translations preserve the `%s` placeholder, while the fine-tuned output is closer in wording to the reference.

### Example 3

**English**

> In my &Start Menu Programs folder

**Reference**

> মোৰ আৰম্ভণি মেনু প্ৰগ্ৰামসমূহ ফোল্ডাৰত (&S)

**Pretrained**

> মোৰ &Start Menu Programs ফোল্ডাৰত

**Fine-tuned**

> মোৰ &Start মেনু প্ৰোগ্ৰাম ফোল্ডাৰত

The fine-tuned model translates more of the English interface phrase into Assamese.

### Example 4

**English**

> Expected , or ) after variable name in variable reference but found %1$S.

The fine-tuned translation preserves the `%1$S` placeholder and more closely follows the structure of the reference than the pretrained output.

## Error Analysis

Fine-tuning improved the aggregate BLEU and chrF scores, but the model is not consistently correct.

One evaluation sentence demonstrates a clear hallucination:

**English**

> Rather than say what might be discouraging, we could strive to be encouraging.

**Reference**

> নিৰন্তৰে ইজনে সিজনক সহায় কৰক

The pretrained model produced an unrelated editing marker, while the fine-tuned model generated unrelated religious content.

This demonstrates that metric improvement does not imply reliable translation on every sentence. The model should therefore be treated as an experimental low-resource NMT system rather than a production translation service.

Other limitations include:

- training uses only a 40,000-pair subset of the available corpus
- only two fine-tuning epochs were used
- evaluation covers 1,000 of the 4,000 held-out test pairs
- some source/reference pairs are noisy or only loosely equivalent
- low-resource translation remains sensitive to domain mismatch and corpus quality

## Reproducibility Artifacts

The repository includes:

- `metrics.csv` — pretrained and fine-tuned BLEU/chrF
- `sample_translations.csv` — qualitative outputs from both models
- `training_log.csv` — loss, validation and runtime history
- `training_time_minutes.txt` — measured training duration

The trained model itself is intentionally not committed because it is approximately 2.4 GB.

The dataset is also not committed; preprocessing downloads it automatically.

## Conclusion

The experiment successfully completed the intended before/after comparison.

Fine-tuning NLLB-200 distilled 600M on 32,000 English–Assamese training pairs improved both evaluation metrics, with BLEU increasing from **7.45 to 9.80** and chrF from **31.82 to 35.29**.

The results demonstrate measurable adaptation to the English–Assamese corpus while also highlighting remaining hallucination and data-quality limitations typical of low-resource machine translation.