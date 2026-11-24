# EXP-8: Abstractive Text Summarization with mT5 and mBART

## Abstract
This experiment compares two multilingual sequence-to-sequence models on abstractive summarization of Hindi news articles from the XL-Sum corpus. A fixed subset of 50 examples from the official test split was used for a fair and reproducible comparison. The models were `csebuetnlp/mT5_multilingual_XLSum` (582.4M parameters) and a locally stored XL-Sum fine-tune of mBART-large-50 (610.9M parameters). Both systems generated summaries with beam search using two beams and at most 80 new tokens. Evaluation used Unicode-aware ROUGE-1, ROUGE-2 and ROUGE-L implemented from scratch, plus greedy-aligned mBERTScore based on `bert-base-multilingual-cased` embeddings. mBART obtained the highest ROUGE-1 score (0.5520), while mT5 led on ROUGE-2 (0.2847), ROUGE-L (0.4017) and mBERTScore (0.6520). Generation time for the 50 articles was 54.4 seconds for mT5 and 83.1 seconds for mBART. The results show a modest quality trade-off in favor of mT5 on this small subset, with a clear speed advantage of about 1.5x.

## Description
The source is a single Jupyter notebook, `EXP-8/exp-8.ipynb`, together with two locally cached model directories under `EXP-8/models/`: `mbart50-xlsum` for the mBART checkpoint and `bert-base-multilingual-cased` for the embedding model used by mBERTScore. The Python stack includes PyTorch 2.10.0, the Hugging Face `transformers` and `datasets` libraries, NumPy, pandas, Matplotlib and Seaborn. All random seeds were fixed to 42, and execution used the Apple MPS backend when available. The XL-Sum Hindi configuration contains 70,778 training, 8,847 validation and 8,847 test articles; only the first 50 test articles were scored.

The pipeline first loads the dataset and separates the article text from the reference summary. ROUGE is implemented directly with a Unicode-aware tokenizer preserving Devanagari characters, because the default `rouge_score` tokenizer would discard Hindi text. ROUGE-N is computed as n-gram F1 overlap and ROUGE-L from a dynamic-programming longest common subsequence. mBERTScore encodes each generated and reference summary with mBERT, normalizes token embeddings, builds a cosine similarity matrix, and averages greedy-alignment precision and recall into an F1 value. Both summarizers are loaded through `AutoModelForSeq2SeqLM`, run in batches of four with `num_beams=2` and `max_new_tokens=80`, and timed with `time.time()`. For mBART the tokenizer source language is explicitly set to `hi_IN`, while mT5 uses the multilingual SentencePiece vocabulary of its XL-Sum checkpoint.

The observed result table reports mT5 at ROUGE-1 0.5444, ROUGE-2 0.2847, ROUGE-L 0.4017, mBERTScore 0.6520 and 54.4 seconds; mBART reached 0.5520, 0.2710, 0.3954, 0.6477 and 83.1 seconds respectively. Per-example generation times were 1.09 s for mT5 and 1.66 s for mBART. Qualitative inspection showed that both models produced fluent but sometimes factually different summaries, for example by focusing on police custody details instead of the reference emphasis on forensic tests. The notebook saves a grouped bar chart comparing all four quality metrics and total generation time to `exp8_comparison.png`. The authors note that neither model is trained in this notebook and that a 50-example subset gives only indicative estimates.

## Code

### Unicode-aware ROUGE implementation
```python
from collections import Counter


def tokenize_rouge(text: str):
    """Unicode-aware tokeniser: keeps letters/digits of any script (incl. Devanagari)."""
    return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)


def _f1(p, r):
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def rouge_n(pred: str, ref: str, n: int) -> float:
    pred_toks, ref_toks = tokenize_rouge(pred), tokenize_rouge(ref)
    if not pred_toks or not ref_toks:
        return 0.0
    pred_ngrams = Counter(
        tuple(pred_toks[i:i + n]) for i in range(len(pred_toks) - n + 1)
    )
    ref_ngrams = Counter(
        tuple(ref_toks[i:i + n]) for i in range(len(ref_toks) - n + 1)
    )
    overlap = sum((pred_ngrams & ref_ngrams).values())
    p = overlap / max(1, sum(pred_ngrams.values()))
    r = overlap / max(1, sum(ref_ngrams.values()))
    return _f1(p, r)


def rouge_l(pred: str, ref: str) -> float:
    pred_toks, ref_toks = tokenize_rouge(pred), tokenize_rouge(ref)
    if not pred_toks or not ref_toks:
        return 0.0
    m, n = len(pred_toks), len(ref_toks)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_toks[i - 1] == ref_toks[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    return _f1(lcs / m, lcs / n)


def compute_rouge(preds, refs):
    scores = {"ROUGE-1": [], "ROUGE-2": [], "ROUGE-L": []}
    for p, r in zip(preds, refs):
        scores["ROUGE-1"].append(rouge_n(p, r, 1))
        scores["ROUGE-2"].append(rouge_n(p, r, 2))
        scores["ROUGE-L"].append(rouge_l(p, r))
    return {k: float(np.mean(v)) for k, v in scores.items()}
```

### mBERTScore with greedy token alignment
```python
MBERT_PATH = "./models/bert-base-multilingual-cased"


def _load_bert(path, device):
    from transformers import AutoModel
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path).to(device).eval()
    return tok, model


def mbert_score(preds, refs, model_path=MBERT_PATH, device=DEVICE):
    """Greedy-aligned mBERT F1 (a.k.a. mBERTScore)."""
    tok, model = _load_bert(model_path, device)
    pred_emb, ref_emb = [], []
    with torch.no_grad():
        for i in range(0, len(preds), 8):
            pc, rc = preds[i:i + 8], refs[i:i + 8]
            pe = tok(pc, return_tensors="pt", padding=True,
                     truncation=True, max_length=256).to(device)
            re_ = tok(rc, return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(device)
            p_out, r_out = model(**pe).last_hidden_state, model(**re_).last_hidden_state
            for j in range(len(pc)):
                pm, rm = pe["attention_mask"][j].bool(), re_["attention_mask"][j].bool()
                pred_emb.append(p_out[j, pm])
                ref_emb.append(r_out[j, rm])
    scores = []
    for p, r in zip(pred_emb, ref_emb):
        p_n = torch.nn.functional.normalize(p, dim=-1)
        r_n = torch.nn.functional.normalize(r, dim=-1)
        sim = p_n @ r_n.T
        prec = sim.max(dim=1).values.mean().item()
        rec = sim.max(dim=0).values.mean().item()
        scores.append(_f1(prec, rec))
    return float(np.mean(scores))
```

### Shared summarization and model-loading functions
```python
MT5_PATH = "csebuetnlp/mT5_multilingual_XLSum"
MBART_PATH = "./models/mbart50-xlsum"
MAX_INPUT_LEN = 512
MAX_NEW_TOKENS = 80
NUM_BEAMS = 2


def load_seq2seq(path, device):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device).eval()
    return tok, model


def summarize(texts, tok, model, device,
              max_input=MAX_INPUT_LEN, max_new=MAX_NEW_TOKENS,
              num_beams=NUM_BEAMS, src_lang=None, batch_size=4):
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        if src_lang is not None:
            tok.src_lang = src_lang
        inputs = tok(batch, return_tensors="pt", padding=True,
                     truncation=True, max_length=max_input).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new,
                                 num_beams=num_beams, early_stopping=True)
        preds.extend(tok.batch_decode(out, skip_special_tokens=True))
    return preds


mt5_tok, mt5_model = load_seq2seq(MT5_PATH, DEVICE)
mbart_tok, mbart_model = load_seq2seq(MBART_PATH, DEVICE)

articles, references = test_ds["text"], test_ds["summary"]
t0 = time.time()
mt5_preds = summarize(articles, mt5_tok, mt5_model, DEVICE)
mt5_time = time.time() - t0
t0 = time.time()
mbart_preds = summarize(articles, mbart_tok, mbart_model, DEVICE, src_lang="hi_IN")
mbart_time = time.time() - t0
```

### Final metric table and comparison chart
```python
mt5_metrics = compute_rouge(mt5_preds, references)
mt5_metrics["mBERTScore"] = mbert_score(mt5_preds, references)
mt5_metrics["Time (s)"] = round(mt5_time, 1)

mbart_metrics = compute_rouge(mbart_preds, references)
mbart_metrics["mBERTScore"] = mbert_score(mbart_preds, references)
mbart_metrics["Time (s)"] = round(mbart_time, 1)

results = pd.DataFrame([mt5_metrics, mbart_metrics], index=["mT5", "mBART"])
results = results[["ROUGE-1", "ROUGE-2", "ROUGE-L", "mBERTScore", "Time (s)"]]
print(results.round(4))

metric_cols = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "mBERTScore"]
plot_df = results[metric_cols].reset_index().melt(
    id_vars="index", var_name="Metric", value_name="Score"
)
plot_df.rename(columns={"index": "Model"}, inplace=True)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", ax=axes[0])
axes[0].set_ylim(0, 1)
axes[0].set_title("Quality metrics: mT5 vs mBART")
time_df = pd.DataFrame({
    "Model": ["mT5", "mBART"],
    "Generation time (s)": [mt5_time, mbart_time],
})
sns.barplot(data=time_df, x="Model", y="Generation time (s)", ax=axes[1])
axes[1].set_title(f"Generation time ({N_EVAL} examples)")
plt.tight_layout()
plt.savefig("exp8_comparison.png", dpi=120, bbox_inches="tight")
plt.show()
```

## Tables

### XL-Sum Hindi dataset size reported by the notebook
| Split | Articles |
| --- | --- |
| Train | 70,778 |
| Validation | 8,847 |
| Test | 8,847 |
| Test subset evaluated | 50 |

### Model sizes and inference settings
| Model | Parameters | Source language setting | Decoding |
| --- | --- | --- | --- |
| mT5 XL-Sum | 582.40128 M | multilingual SentencePiece | beam=2, max_new_tokens=80 |
| mBART-large-50 XL-Sum | 610.879488 M | `hi_IN` | beam=2, max_new_tokens=80 |

### Observed evaluation results on 50 Hindi test examples
| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | mBERTScore | Time (s) | Seconds/example |
| --- | --- | --- | --- | --- | --- | --- |
| mT5 | 0.5444 | 0.2847 | 0.4017 | 0.6520 | 54.4 | 1.09 |
| mBART | 0.5520 | 0.2710 | 0.3954 | 0.6477 | 83.1 | 1.66 |

### Observed differences (mT5 minus mBART)
| ROUGE-1 | ROUGE-2 | ROUGE-L | mBERTScore | Time (s) |
| --- | --- | --- | --- | --- |
| -0.0076 | +0.0137 | +0.0063 | +0.0043 | -28.7 |

## Figures

| path | caption |
| --- | --- |
| /Users/kartiken/Downloads/CLNLP-LAB/EXP-8/exp8_comparison.png | Grouped bar charts of ROUGE-1, ROUGE-2, ROUGE-L and mBERTScore for mT5 versus mBART, alongside total generation time for 50 Hindi XL-Sum test articles. |
