# 5. Subword Tokenization and POS Tagging

## Abstract
This experiment compares subword tokenization and part-of-speech tagging across pretrained and corpus-trained models. The corpus file `input_sub_word_data.txt` contains 6,682 characters, 122 lines and 958 lowercase alphabetic words, while SentencePiece training reported 6,661 normalized characters, 99.97% character coverage and an alphabet of 44. A GPT-2 BPE tokenizer split the long-word sentence into 31 tokens, whereas 200 BPE merges learned from the corpus produced a 205-symbol vocabulary and 69 tokens for the same sentence. ALBERT SentencePiece produced 34 tokens, and a unigram SentencePiece model trained with vocabulary size 400 produced 49 tokens. POS tagging of the sample sentence returned 12 unique tokens under both spaCy and NLTK with corresponding grammatical roles, and the corpus-level spaCy table contained 395 unique tokens. The most frequent corpus tokens were `,` (76), `.` (65), `the` (39), `and` (27) and `can` (23), and a final regex step extracted two phone-number matches and one date.

## Description
The notebook uses `input_sub_word_data.txt` as the shared corpus for BPE and SentencePiece training and uses a fixed sentence containing long derived words for tokenizer comparison. Libraries include `transformers`, `re`, `collections.Counter` and `defaultdict`, `sentencepiece`, `spacy`, `nltk` and `pandas`. The saved artifacts `corpus_sentencepiece.model` and `corpus_sentencepiece.vocab` contain a 400-entry unigram vocabulary; the trainer log reports 122 loaded sentences, 103 preprocessed sentences, 6,661 characters, 99.97% coverage, alphabet size 44 and final vocabulary size 400.

For BPE, a pretrained `GPT2Tokenizer` tokenized the long-word sentence into 31 pieces and IDs, splitting `naturalization` into `natural` (11802), `ization` (1634) and `,` (11), and `computationally` into `Ġcomput` (2653) and `ationally` (15208). A from-scratch BPE pipeline lowercased the corpus, represented every word as spaced characters plus `</w>`, counted adjacent-symbol pairs, performed 200 greedy merges and built a 205-symbol vocabulary. Applying those merges to the same sentence gave 69 tokens, beginning with `nat` (204), `ur` (154), `al` (73) and `ization</w>` (186).

For SentencePiece, `AlbertTokenizer.from_pretrained("albert-base-v2")` returned 34 tokens, including `▁natural` (1112), `ization` (1829) and `,` (15). A locally trained unigram SentencePiece processor returned 49 tokens, including `▁natural` (59), `ization` (31) and `,` (4); the longer piece list shows `▁preprocessing` (115) as a single token and `hydrokinetic` split into `▁h`, `y`, `dr`, `ok`, `ine` and `tic`.

POS tagging used the sentence "The young student is reading an interesting book in the library." spaCy produced 12 unique rows with tags DET, ADJ, NOUN, AUX, VERB, ADP, PUNCT and descriptions such as "auxiliary" and "adposition"; NLTK's UPenn tagset produced 12 corresponding rows such as DT, JJ, NN, VBZ, VBG and IN. Applying spaCy to the full corpus produced 395 unique-token rows sorted by frequency, headed by comma 76, period 65, `the` 39, `and` 27 and `can` 23. Finally, regex patterns over the string `+91-9653433212 12-12-2005 6789012345` returned `['+91-9653433212', '6789012345']` and `['12-12-2005']`.

## Code
### Pretrained GPT-2 BPE and corpus preparation for scratch BPE
```python
from transformers import GPT2Tokenizer
import pandas as pd

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

sentence = (
    "naturalization, internationalization, computationally, unbelievable, "
    "preprocessing, classification, communication, renewable, hydrokinetic, "
    "multilingual, understanding, and modernization"
)

tokens = gpt2_tokenizer.tokenize(sentence)
token_ids = gpt2_tokenizer.convert_tokens_to_ids(tokens)

df_bpe_pretrained = pd.DataFrame({"Token": tokens, "Token ID": token_ids})
df_bpe_pretrained

import re
from collections import Counter, defaultdict

file_path = "input_sub_word_data.txt"
with open(file_path, "r", encoding="utf-8") as f:
    corpus_text = f.read()

words = re.findall(r"[a-z]+", corpus_text.lower())
word_freqs = Counter(" ".join(list(word)) + " </w>" for word in words)
```

### BPE merge learning and tokenization from scratch
```python
def get_pair_freqs(word_freqs):
    pairs = defaultdict(int)
    for word, freq in word_freqs.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def merge_pair(pair, word_freqs):
    bigram = re.escape(" ".join(pair))
    pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    return {pattern.sub("".join(pair), word): freq for word, freq in word_freqs.items()}

num_merges = 200
merges = []

for _ in range(num_merges):
    pair_freqs = get_pair_freqs(word_freqs)
    if not pair_freqs:
        break
    best_pair = max(pair_freqs, key=pair_freqs.get)
    word_freqs = merge_pair(best_pair, word_freqs)
    merges.append(best_pair)

vocab = {}
for word in word_freqs:
    for symbol in word.split():
        vocab.setdefault(symbol, len(vocab))

len(merges), len(vocab)

def bpe_tokenize_word(word, merges):
    symbols = list(word.lower()) + ["</w>"]
    for pair in merges:
        i = 0
        new_symbols = []
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                new_symbols.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols
    return symbols

words_to_tokenize = re.findall(r"[a-zA-Z]+", sentence)

bpe_tokens, bpe_ids = [], []
for word in words_to_tokenize:
    for token in bpe_tokenize_word(word, merges):
        bpe_tokens.append(token)
        bpe_ids.append(vocab.get(token, -1))

df_bpe_scratch = pd.DataFrame({"Token": bpe_tokens, "Token ID": bpe_ids})
df_bpe_scratch
```

### Pretrained ALBERT and locally trained SentencePiece
```python
from transformers import AlbertTokenizer

sp_pretrained_tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
sp_tokens = sp_pretrained_tokenizer.tokenize(sentence)
sp_ids = sp_pretrained_tokenizer.convert_tokens_to_ids(sp_tokens)

df_sp_pretrained = pd.DataFrame({"Token": sp_tokens, "Token ID": sp_ids})
df_sp_pretrained

import os
import sentencepiece as spm

with open(os.devnull, "w") as log_file:
    spm.SentencePieceTrainer.train(
        input=file_path,
        model_prefix="corpus_sentencepiece",
        vocab_size=400,
        model_type="unigram",
        logstream=log_file,
    )

sp_scratch = spm.SentencePieceProcessor(model_file="corpus_sentencepiece.model")
scratch_tokens = sp_scratch.encode(sentence, out_type=str)
scratch_ids = sp_scratch.encode(sentence, out_type=int)

df_sp_scratch = pd.DataFrame({"Token": scratch_tokens, "Token ID": scratch_ids})
df_sp_scratch
```

### POS tagging with spaCy, NLTK and corpus frequencies
```python
import spacy

nlp = spacy.load("en_core_web_sm")
pos_sentence = "The young student is reading an interesting book in the library."
doc = nlp(pos_sentence)

seen_tokens = set()
spacy_rows = []
for token in doc:
    if token.is_space or token.text in seen_tokens:
        continue
    seen_tokens.add(token.text)
    spacy_rows.append((token.text, token.pos_, spacy.explain(token.pos_)))

df_pos_spacy = pd.DataFrame(spacy_rows, columns=["Token", "POS Tag", "Description"])
df_pos_spacy

import nltk

nltk_tokens = nltk.word_tokenize(pos_sentence)
nltk_tagged = nltk.pos_tag(nltk_tokens)
tag_descriptions = dict(nltk.data.load("help/tagsets/upenn_tagset.pickle"))

seen_tokens = set()
nltk_rows = []
for word, tag in nltk_tagged:
    if word in seen_tokens:
        continue
    seen_tokens.add(word)
    description = tag_descriptions.get(tag, (tag,))[0]
    nltk_rows.append((word, tag, description))

df_pos_nltk = pd.DataFrame(nltk_rows, columns=["Token", "POS Tag", "Description"])
df_pos_nltk

corpus_doc = nlp(corpus_text)
corpus_tokens = [token for token in corpus_doc if not token.is_space]

token_freq = Counter(token.text for token in corpus_tokens)

first_occurrence = {}
for token in corpus_tokens:
    first_occurrence.setdefault(token.text, token)

pos_freq_rows = [
    (text, tok.pos_, spacy.explain(tok.pos_), token_freq[text])
    for text, tok in first_occurrence.items()
]

df_pos_freq = pd.DataFrame(
    pos_freq_rows, columns=["Token", "POS Tag", "Description", "Frequency"]
)
df_pos_freq = df_pos_freq.sort_values("Frequency", ascending=False).reset_index(drop=True)
df_pos_freq
```

## Tables
### Token counts for the long-word sentence
| Tokenizer | Source | Token count |
| --- | --- | --- |
| GPT-2 BPE | pretrained `gpt2` | 31 |
| BPE from scratch | 200 merges, 205 symbols | 69 |
| ALBERT SentencePiece | pretrained `albert-base-v2` | 34 |
| SentencePiece unigram | `corpus_sentencepiece.model`, 400 symbols | 49 |

### Example subword tokens and IDs
| Tokenizer | First observed tokens and IDs |
| --- | --- |
| GPT-2 BPE | natural 11802; ization 1634; , 11 |
| BPE from scratch | nat 204; ur 154; al 73; ization</w> 186 |
| ALBERT SentencePiece | ▁natural 1112; ization 1829; , 15 |
| SentencePiece unigram | ▁natural 59; ization 31; , 4 |

### POS tag comparison for the sample sentence
| Token | spaCy POS | NLTK tag |
| --- | --- | --- |
| The | DET | DT |
| young | ADJ | JJ |
| student | NOUN | NN |
| is | AUX | VBZ |
| reading | VERB | VBG |
| an | DET | DT |
| interesting | ADJ | JJ |
| book | NOUN | NN |
| in | ADP | IN |
| the | DET | DT |
| library | NOUN | NN |
| . | PUNCT | . |

### Most frequent corpus tokens from spaCy
| Token | POS Tag | Description | Frequency |
| --- | --- | --- | --- |
| , | PUNCT | punctuation | 76 |
| . | PUNCT | punctuation | 65 |
| the | DET | determiner | 39 |
| and | CCONJ | coordinating conjunction | 27 |
| can | AUX | auxiliary | 23 |

### Regex extraction results
| Field | Matches |
| --- | --- |
| Phone numbers | +91-9653433212, 6789012345 |
| Date of birth | 12-12-2005 |

## Figures
| path | caption |
| --- | --- |
| None | |
