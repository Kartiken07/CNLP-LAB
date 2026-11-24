# EXP-10: Low-Resource Neural Machine Translation: English to Assamese

## Abstract
This experiment fine-tunes `facebook/nllb-200-distilled-600M` for low-resource English-to-Assamese neural machine translation using the AI4Bharat Samanantar parallel corpus. The raw corpus contained 141,227 sentence pairs; cleaning, length filtering, length-ratio filtering and deduplication left 139,707 pairs, from which 40,000 were selected with seed 42 and split into 32,000 training, 4,000 validation and 4,000 test pairs. Training used two epochs, batch size 4, gradient accumulation 4 for an effective batch size of 16, learning rate 2e-5, FP16 precision and a maximum sequence length of 128 on a Kaggle NVIDIA Tesla T4 GPU. The run completed 4,000 optimization steps in 54.3 minutes and reached a final validation loss of 1.9105. Both the pretrained baseline and the fine-tuned checkpoint were scored with SacreBLEU on the same 1,000 held-out test pairs. Fine-tuning raised BLEU from 7.45 to 9.80 and chrF from 31.82 to 35.29, corresponding to gains of 2.35 and 3.47 points. Qualitative examples show improved terminology and placeholder preservation, but also occasional hallucination, so the system remains an experimental translation model rather than a production service.

## Description
The repository has three Kaggle notebooks, `notebooks/exp10_nmt_kaggle.ipynb`, `notebooks/run_all_kaggle.ipynb` and `notebooks/show_results.ipynb`, plus three reusable scripts in `src/`: `preprocess.py`, `train.py` and `evaluate.py`. The README, `results/experiment_summary.md`, `results/metrics.csv`, `results/training_log.csv`, `results/sample_translations.csv` and `results/training_time_minutes.txt` document and store the real outputs of the Kaggle run. The stack uses Python 3.10+, PyTorch, Hugging Face Transformers and Datasets, SentencePiece, Accelerate, SacreBLEU and pandas. The model and dataset are licensed for academic or non-commercial use and are intentionally excluded from version control; the fine-tuned checkpoint is approximately 2.4 GB.

Preprocessing normalizes Unicode to NFC, removes control and invisible characters, collapses whitespace, removes empty or overlong sentences, filters extreme source/target length ratios above 4, removes duplicate pairs, deterministically subsamples 40,000 pairs and splits them 80:10:10. The observed counts were 141,227 raw pairs, 141,227 after empty removal, 141,206 after length filtering, 139,713 after length-ratio filtering, 139,707 after deduplication and 40,000 selected pairs. Training tokenizes with source code `eng_Latn` and target code `asm_Beng`, uses `Seq2SeqTrainer` with validation after every epoch, keeps the best checkpoint by validation loss, and logs FP16 training on CUDA. The training log records loss falling from 11.9262 at step 50 to 7.7227 at step 4,000, with validation loss 1.942178 after epoch 1 and 1.910485 after epoch 2.

Evaluation generates translations with beam search width 4 and a maximum of 128 new tokens, forces the target language token, and computes corpus BLEU and chrF with SacreBLEU. The pretrained baseline scored 7.45 BLEU and 31.82 chrF, while the fine-tuned model scored 9.80 BLEU and 35.29 chrF on the same 1,000 test pairs. The relative improvements were about 31.5% for BLEU and 10.9% for chrF. Qualitative analysis found that the fine-tuned model used terminology closer to the reference for a sentence about public servants and translated more of an interface phrase, while both models preserved placeholders such as `%s` and `%1$S`. In one documented failure case the pretrained model emitted an unrelated editing marker and the fine-tuned model produced unrelated content, demonstrating that aggregate metric gains do not guarantee sentence-level reliability.

## Code

### Parallel-data cleaning and deterministic splitting
```python
SEED = 42


def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def clean_pairs(df, min_chars=1, max_chars=1000, max_len_ratio=4.0):
    """Return (clean_df, stats). Removes empty, missing, duplicate and badly misaligned pairs."""
    stats = {"raw": len(df)}
    df = df.copy()
    df["src"] = df["src"].map(clean_text)
    df["tgt"] = df["tgt"].map(clean_text)
    df = df[(df.src.str.len() >= min_chars) & (df.tgt.str.len() >= min_chars)]
    stats["after_empty_removal"] = len(df)
    df = df[(df.src.str.len() <= max_chars) & (df.tgt.str.len() <= max_chars)]
    stats["after_length_filter"] = len(df)
    ratio = (df.src.str.len() / df.tgt.str.len()).clip(lower=1e-9)
    df = df[(ratio <= max_len_ratio) & (ratio >= 1 / max_len_ratio)]
    stats["after_ratio_filter"] = len(df)
    df = df.drop_duplicates(subset=["src", "tgt"]).reset_index(drop=True)
    stats["after_dedup"] = len(df)
    return df, stats


def split_pairs(df, seed=SEED, train=0.8, val=0.1):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train, n_val = int(n * train), int(n * val)
    return df[:n_train], df[n_train:n_train + n_val], df[n_train + n_val:]


MAX_ROWS = 40000
df, stats = clean_pairs(load_hf("samanantar"))
if len(df) > MAX_ROWS:
    df = df.sample(n=MAX_ROWS, random_state=SEED).reset_index(drop=True)
tr, va, te = split_pairs(df, seed=SEED)
print(stats, {"train": len(tr), "validation": len(va), "test": len(te)})
```

### NLLB fine-tuning configuration and trainer
```python
LANG_CODES = {"en": "eng_Latn", "as": "asm_Beng"}
set_seed(42)
tokenizer = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-600M",
    src_lang=LANG_CODES["en"], tgt_lang=LANG_CODES["as"]
)
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

targs = Seq2SeqTrainingArguments(
    output_dir="nllb-en-as",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=0.03,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    seed=42,
    report_to="none",
)
trainer = Seq2SeqTrainer(
    model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    processing_class=tokenizer,
)
t0 = time.time()
trainer.train()
minutes = (time.time() - t0) / 60
trainer.save_model("nllb-en-as")
pd.DataFrame(trainer.state.log_history).to_csv("results/training_log.csv", index=False)
print(f"Training time: {minutes:.1f} min")
```

### BLEU and chrF evaluation on identical test examples
```python
LANG_CODES = {"en": "eng_Latn", "as": "asm_Beng"}


def translate(model, tokenizer, sentences, tgt_code, batch_size, max_length, num_beams, device):
    out = []
    forced = tokenizer.convert_tokens_to_ids(tgt_code)
    for i in range(0, len(sentences), batch_size):
        batch = tokenizer(
            sentences[i:i + batch_size], return_tensors="pt", padding=True,
            truncation=True, max_length=max_length
        ).to(device)
        with torch.no_grad():
            gen = model.generate(
                **batch, forced_bos_token_id=forced,
                max_new_tokens=max_length, num_beams=num_beams
            )
        out += tokenizer.batch_decode(gen, skip_special_tokens=True)
    return out


df = pd.read_csv("data/processed/test.csv", dtype=str, keep_default_na=False)
df = df.head(1000)
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(
    args.model, src_lang=LANG_CODES["en"], tgt_lang=LANG_CODES["as"]
)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()
hyps = translate(model, tokenizer, df.src.tolist(), LANG_CODES["as"], 16, 128, 4, device)
refs = df.tgt.tolist()
bleu = sacrebleu.corpus_bleu(hyps, [refs])
chrf = sacrebleu.corpus_chrf(hyps, [refs])
print(f"{args.label}: BLEU={bleu.score:.2f} chrF={chrf.score:.2f} on {len(df)} test pairs")
```

### Interactive translation demo from the Kaggle notebook
```python
import torch
from transformers import AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained(
    'nllb-en-as', src_lang='eng_Latn', tgt_lang='asm_Beng'
)
model = AutoModelForSeq2SeqLM.from_pretrained('nllb-en-as')
model = model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()


def translate(text):
    b = tok(text, return_tensors='pt', truncation=True, max_length=128).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **b,
            forced_bos_token_id=tok.convert_tokens_to_ids('asm_Beng'),
            max_new_tokens=128,
            num_beams=4
        )
    return tok.batch_decode(out, skip_special_tokens=True)[0]


while True:
    s = input('English sentence (blank to stop): ').strip()
    if not s:
        break
    print('Assamese:', translate(s))
```

## Tables

### Observed BLEU and chrF from `results/metrics.csv`
| Model | Direction | Test pairs | BLEU | chrF |
| --- | --- | ---: | ---: | ---: |
| Pretrained (before fine-tuning) | en-as | 1,000 | 7.45 | 31.82 |
| Fine-tuned | en-as | 1,000 | 9.80 | 35.29 |

### Improvement after fine-tuning
| Metric | Before | After | Absolute gain | Relative gain |
| --- | ---: | ---: | ---: | ---: |
| BLEU | 7.45 | 9.80 | +2.35 | about 31.5% |
| chrF | 31.82 | 35.29 | +3.47 | about 10.9% |

### Observed preprocessing counts
| Stage | Pairs |
| --- | ---: |
| Raw Samanantar Assamese configuration | 141,227 |
| After empty removal | 141,227 |
| After length filtering | 141,206 |
| After length-ratio filtering | 139,713 |
| After deduplication | 139,707 |
| Selected for experiment | 40,000 |

### Final deterministic split
| Split | Pairs |
| --- | ---: |
| Training | 32,000 |
| Validation | 4,000 |
| Test | 4,000 |
| Evaluated test subset | 1,000 |

### Experiment configuration from `experiment_summary.md`
| Setting | Value |
| --- | --- |
| Base model | `facebook/nllb-200-distilled-600M` |
| Direction | English to Assamese (`en-as`) |
| Source / target codes | `eng_Latn` / `asm_Beng` |
| Epochs | 2 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Effective batch size | 16 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Maximum sequence length | 128 |
| Random seed | 42 |
| Mixed precision | FP16 |
| Best-model criterion | Validation loss |
| GPU | NVIDIA Tesla T4 |
| Training time | 54.3 minutes |

### Observed training and validation statistics
| Statistic | Value |
| --- | --- |
| Optimization steps | 4,000 |
| Loss at step 50 | 11.926239 |
| Loss at step 2,000 | 8.119777 |
| Loss at step 4,000 | 7.722725 |
| Validation loss after epoch 1 | 1.942178 |
| Validation loss after epoch 2 | 1.910485 |
| Final recorded validation loss | about 1.9105 |
| Training runtime | 3,259.5557 seconds |
| Training samples per second | 19.635 |
| Training steps per second | 1.227 |
| Total floating-point operations | 4.592526e15 |
| Final recorded training loss | 8.272694 |

### Qualitative observations in English
| English source | Observed outcome |
| --- | --- |
| We are public servants. | Fine-tuned output uses terminology closer to the reference. |
| Log out %s of this session to log in as a different user | Both models preserve the `%s` placeholder; fine-tuned wording is closer to the reference. |
| In my &Start Menu Programs folder | Fine-tuned output translates more of the interface phrase into Assamese. |
| Expected , or ) after variable name in variable reference but found %1$S. | Fine-tuned output preserves `%1$S` and follows the reference structure more closely. |
| Rather than say what might be discouraging, we could strive to be encouraging. | Baseline emitted an unrelated editing marker; fine-tuned model generated unrelated content. |

## Figures

| path | caption |
| --- | --- |
| None | |
