# 6. GloVe (Global Vectors for Word Representation) and N-gram Statistical Language Model

## Abstract
This experiment implements GloVe embeddings from scratch and compares them with pretrained Gensim vectors, then builds an N-gram statistical language model from `text_data.txt`. The GloVe corpus contains 24 sentences and 68 unique lowercase words, producing a 68 by 68 co-occurrence matrix with 526 non-zero word-context pairs. Training used 50 dimensions, `x_max` 100.0, `alpha` 0.75, initial learning rate 0.05, 6,000 epochs and seed 42, with loss falling from 8.800191 at epoch 1 to 0.000027 at epoch 6000. For `king`, the scratch model's top neighbors are `the` (0.8189), `lives` (0.7325) and `queen` (0.5854), while `glove-wiki-gigaword-50` gives `prince` (0.8236), `queen` (0.7839) and `ii` (0.7746). The companion N-gram model uses 8 sentences and a 36-token vocabulary including boundary markers, with 35, 53 and 61 unique unigrams, bigrams and trigrams. Corpus perplexity is 27.584 for unigrams, 15.223 for bigrams and 16.478 for trigrams, and greedy bigram decoding completes `natural language` as `natural language processing enables computers to understand human language processing`.

## Description
Two notebooks cover this experiment: `exp-6.ipynb` implements GloVe, and `exp-6-2.ipynb` implements a statistical N-gram language model. The GloVe notebook reads `glove_corpus.txt`, tokenizes with `re.findall(r"[a-z]+", text.lower())`, builds a sorted vocabulary of 68 words and uses a symmetric window of 5 to accumulate weighted co-occurrences; pairs closer in the sentence receive larger weights such as `1.0 / abs(i - j)`. The resulting matrix has shape `(68, 68)` and 526 non-zero word-context pairs. Libraries are `re`, `numpy`, `pandas` and `gensim.downloader`.

The from-scratch GloVe optimizer initializes word and context vectors with `np.random.default_rng(42).normal(scale=0.05)`, adds word and context biases, applies the weighting function `(x / x_max) ** alpha` below `x_max` and 1.0 otherwise, and uses a linearly decaying learning rate. It prints losses at epoch 1 and every 1,000 epochs: 8.800191, 0.017741, 0.001021, 0.000182, 0.000062, 0.000033 and 0.000027. Cosine similarity over the summed word and context vectors ranks neighbors of `king`; the small corpus favors local co-occurrences such as `the`, `lives`, `queen`, `is`, `rules`, `kingdom`, `princess` and `prince`.

The pretrained comparison loads `glove-wiki-gigaword-50` through Gensim. Its nearest neighbors of `king` are `prince` 0.8236, `queen` 0.7839, `ii` 0.7746, `emperor` 0.7736, `son` 0.7667, `uncle` 0.7627, `kingdom` 0.7542, `throne` 0.7540, `brother` 0.7492 and `ruler` 0.7434. This illustrates the notebook observation that large Wikipedia and Gigaword statistics yield more semantic neighbors than the tiny local corpus.

The N-gram notebook reads 8 lines from `text_data.txt`, pads sentences with `<s>` and `</s>`, and counts unigrams, bigrams and trigrams with add-one Laplace smoothing over a vocabulary of 36. It reports 35, 53 and 61 unique n-grams for orders 1, 2 and 3. For the context `natural language`, the top bigram continuation is `processing` (0.093) and the top trigram continuation is `processing` (0.103); for `is a`, the top bigram and trigram continuation is `branch` (0.073 and 0.077). Greedy bigram completion produces `natural language processing enables computers to understand human language processing`, `machine learning` and `a language model predicts the next word in a branch of`. Perplexity over the corpus is 27.584, 15.223 and 16.478 for the three model orders, so the bigram model fits this corpus best.

## Code
### GloVe corpus, vocabulary and weighted co-occurrence matrix
```python
import re
import numpy as np
import pandas as pd

with open("glove_corpus.txt", "r", encoding="utf-8") as f:
    sentences = f.read().splitlines()

def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())

tokens = [tokenize(sentence) for sentence in sentences]
vocab = sorted({word for sentence in tokens for word in sentence})
word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for word, i in word2idx.items()}
V = len(vocab)

print(f"Sentences: {len(sentences)}")
print(f"Vocabulary size: {V}")
print(vocab)

WINDOW = 5
cooccurrence = np.zeros((V, V), dtype=np.float64)

for sentence in tokens:
    indices = [word2idx[word] for word in sentence]
    for i, wi in enumerate(indices):
        for j in range(max(0, i - WINDOW), min(len(indices), i + WINDOW + 1)):
            if i == j:
                continue
            cooccurrence[wi, indices[j]] += 1.0 / abs(i - j)

print(f"Co-occurrence matrix shape: {cooccurrence.shape}")
print(f"Non-zero (word, context) pairs: {int((cooccurrence > 0).sum())}")
```

### GloVe training, scratch similarities and pretrained comparison
```python
def glove(cooccurrence, dim=50, x_max=100.0, alpha=0.75, lr0=0.05, epochs=6000, seed=42):
    rng = np.random.default_rng(seed)
    X = cooccurrence
    mask = X > 0
    log_X = np.zeros_like(X)
    log_X[mask] = np.log(X[mask])
    weights = np.where(mask, np.where(X < x_max, (X / x_max) ** alpha, 1.0), 0.0)
    W = rng.normal(scale=0.05, size=(V, dim))
    W_context = rng.normal(scale=0.05, size=(V, dim))
    b_word = np.zeros(V)
    b_context = np.zeros(V)
    for epoch in range(1, epochs + 1):
        pred = W @ W_context.T + b_word[:, None] + b_context[None, :]
        diff = np.where(mask, pred - log_X, 0.0)
        g = 2.0 * weights * diff
        lr = lr0 * max(0.0, 1.0 - epoch / epochs)
        W -= lr * (g @ W_context)
        W_context -= lr * (g.T @ W)
        b_word -= lr * g.sum(axis=1)
        b_context -= lr * g.sum(axis=0)
        if epoch % 1000 == 0 or epoch == 1:
            loss = float(np.sum(weights * diff ** 2))
            print(f"Epoch {epoch:5d} | Loss {loss:.6f}")
    embeddings = W + W_context
    return embeddings

embeddings = glove(cooccurrence)
def most_similar(word, embeddings, topn=10):
    vec = embeddings[word2idx[word]]
    norms = np.linalg.norm(embeddings, axis=1) * (np.linalg.norm(vec) + 1e-9)
    sims = embeddings @ vec / norms
    order = np.argsort(-sims)
    rows = []
    for i in order:
        if idx2word[i] == word:
            continue
        rows.append({"Word": idx2word[i], "Similarity": round(float(sims[i]), 4)})
        if len(rows) == topn:
            break
    return pd.DataFrame(rows)

df_king_scratch = most_similar("king", embeddings)
df_king_scratch

import gensim.downloader as api
glove_pretrained = api.load("glove-wiki-gigaword-50")
df_king_pretrained = pd.DataFrame(
    [
        {"Word": word, "Similarity": round(float(score), 4)}
        for word, score in glove_pretrained.most_similar("king", topn=10)
    ]
)
df_king_pretrained
```

### N-gram counts, smoothed probabilities and top-k prediction
```python
import re
import math
from collections import Counter
with open("text_data.txt", "r", encoding="utf-8") as f:
    sentences = f.read().splitlines()
def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())
tokenized = [tokenize(s) for s in sentences]
def build_ngram_model(tokenized_corpus, n, alpha=1.0):
    ngram_counts = Counter()
    context_counts = Counter()
    for sentence in tokenized_corpus:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ngram = tuple(padded[i : i + n])
            ngram_counts[ngram] += 1
            context_counts[ngram[:-1]] += 1
    vocab = sorted({w for sent in tokenized_corpus for w in sent} | {"<s>", "</s>"})
    V = len(vocab)
    return ngram_counts, context_counts, vocab, V

def prob(word, context, ngram_counts, context_counts, vocab, V, alpha=1.0):
    count_ctx = context_counts[context]
    return (ngram_counts[context + (word,)] + alpha) / (count_ctx + alpha * V)

models = {}
for n in (1, 2, 3):
    models[n] = build_ngram_model(tokenized, n)

for n, (ng, ctx, vocab, V) in models.items():
    print(f"{n}-gram: {len(ng)} unique n-grams, vocab size = {V}")

def predict_next(context_words, n, topk=5):
    ng, ctx_counts, vocab, V = models[n]
    if n == 1:
        context = ()
    else:
        needed = n - 1
        prefix = ["<s>"] * max(0, needed - len(context_words))
        context = tuple(prefix + context_words[-needed:])
    scores = []
    for w in vocab:
        if w == "<s>":
            continue
        scores.append((w, prob(w, context, ng, ctx_counts, vocab, V)))
    scores.sort(key=lambda x: -x[1])
    return scores[:topk]

def show_predictions(context_words):
    print(f"Context: {context_words}")
    for n in (1, 2, 3):
        label = {1: "Unigram", 2: "Bigram ", 3: "Trigram"}[n]
        preds = predict_next(context_words, n)
        formatted = ", ".join(f"{w} ({p:.3f})" for w, p in preds)
        print(f"  {label} -> {formatted}")
    print()
for ctx in [["natural", "language"], ["machine", "learning"], ["is", "a"], ["language", "models"], ["the"]]:
    show_predictions(ctx)
```

### Greedy sentence completion and corpus perplexity
```python
def complete_sentence(prefix_words, n=2, max_words=8):
    ng, ctx_counts, vocab, V = models[n]
    generated = list(prefix_words)
    for _ in range(max_words):
        context = tuple(generated[-(n - 1):]) if n > 1 else ()
        best_word, best_p = None, -1.0
        for w in vocab:
            if w == "<s>":
                continue
            p = prob(w, context, ng, ctx_counts, vocab, V)
            if p > best_p:
                best_word, best_p = w, p
        if best_word == "</s>":
            break
        generated.append(best_word)
    return generated

for prefix in [["natural", "language"], ["machine", "learning"], ["a", "language", "model"]]:
    print(" ".join(prefix), "->", " ".join(complete_sentence(prefix)))

def sentence_log_prob(sentence, n):
    ng, ctx_counts, vocab, V = models[n]
    padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
    log_p = 0.0
    for i in range(len(padded) - n + 1):
        context = tuple(padded[i : i + n - 1])
        word = padded[i + n - 1]
        p = prob(word, context, ng, ctx_counts, vocab, V)
        log_p += math.log(p)
    return log_p

def perplexity(corpus, n):
    total_log_p = sum(sentence_log_prob(s, n) for s in corpus)
    total_tokens = sum(len(s) + 1 for s in corpus)
    return math.exp(-total_log_p / total_tokens)

print(f"{'Model':<10} {'Perplexity':>12}")
print("-" * 24)
for n in (1, 2, 3):
    print(f"{n}-gram{'':<5} {perplexity(tokenized, n):>12.3f}")
```

## Tables
### GloVe corpus and training configuration
| Metric | Value |
| --- | --- |
| Sentences in `glove_corpus.txt` | 24 |
| Vocabulary size | 68 |
| Co-occurrence matrix shape | (68, 68) |
| Non-zero word-context pairs | 526 |
| Embedding dimension | 50 |
| `x_max` / `alpha` | 100.0 / 0.75 |
| Initial learning rate / epochs / seed | 0.05 / 6000 / 42 |

### GloVe training loss by epoch
| Epoch | Loss |
| --- | --- |
| 1 | 8.800191 |
| 1000 | 0.017741 |
| 2000 | 0.001021 |
| 3000 | 0.000182 |
| 4000 | 0.000062 |
| 5000 | 0.000033 |
| 6000 | 0.000027 |

### Most similar words to `king`
| Rank | Scratch word | Scratch similarity | Pretrained word | Pretrained similarity |
| --- | --- | --- | --- | --- |
| 1 | the | 0.8189 | prince | 0.8236 |
| 2 | lives | 0.7325 | queen | 0.7839 |
| 3 | queen | 0.5854 | ii | 0.7746 |
| 4 | is | 0.5750 | emperor | 0.7736 |
| 5 | rules | 0.4830 | son | 0.7667 |
| 6 | and | 0.3541 | uncle | 0.7627 |
| 7 | kingdom | 0.3497 | kingdom | 0.7542 |
| 8 | princess | 0.3381 | throne | 0.7540 |
| 9 | prince | 0.3296 | brother | 0.7492 |
| 10 | sun | 0.3019 | ruler | 0.7434 |

### N-gram model statistics and corpus perplexity
| Model | Unique n-grams | Vocabulary size | Perplexity | Top continuation for `natural language` |
| --- | --- | --- | --- | --- |
| 1-gram | 35 | 36 | 27.584 | </s> (0.083) |
| 2-gram | 53 | 36 | 15.223 | processing (0.093) |
| 3-gram | 61 | 36 | 16.478 | processing (0.103) |

### Top five next-word predictions by context
| Context | Unigram | Bigram | Trigram |
| --- | --- | --- | --- |
| natural language | </s> 0.083, language 0.073, a 0.055, learning 0.055, artificial 0.037 | processing 0.093, models 0.070, </s> 0.047, model 0.047, a 0.023 | processing 0.103, </s> 0.026, a 0.026, and 0.026, artificial 0.026 |
| machine learning | </s> 0.083, language 0.073, a 0.055, learning 0.055, artificial 0.037 | </s> 0.073, is 0.073, and 0.049, a 0.024, artificial 0.024 | </s> 0.051, and 0.051, is 0.051, a 0.026, artificial 0.026 |
| is a | </s> 0.083, language 0.073, a 0.055, learning 0.055, artificial 0.037 | branch 0.073, language 0.049, part 0.049, sentence 0.049, </s> 0.024 | branch 0.077, part 0.051, </s> 0.026, a 0.026, and 0.026 |
| language models | </s> 0.083, language 0.073, a 0.055, learning 0.055, artificial 0.037 | </s> 0.053, learn 0.053, a 0.026, and 0.026, artificial 0.026 | </s> 0.053, learn 0.053, a 0.026, and 0.026, artificial 0.026 |
| the | </s> 0.083, language 0.073, a 0.055, learning 0.055, artificial 0.037 | next 0.054, </s> 0.027, a 0.027, and 0.027, artificial 0.027 | </s> 0.028, a 0.028, and 0.028, artificial 0.028, branch 0.028 |

### Greedy bigram completions
| Prefix | Generated sequence |
| --- | --- |
| natural language | natural language processing enables computers to understand human language processing |
| machine learning | machine learning |
| a language model | a language model predicts the next word in a branch of |

## Figures
| path | caption |
| --- | --- |
| None | |
