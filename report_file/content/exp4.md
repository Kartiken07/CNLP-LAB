# 4. Term Frequency and NER

## Abstract
This experiment analyzes term frequencies and named entities in a 17,345-character, 67-line corpus stored in `4.1_4.2_input.txt`. Regex tokenization with `\b\w+\b` produced 2,446 tokens and 749 unique terms, while simple whitespace splitting produced 2,419 tokens; both rankings begin with `a` (90), `and` (86) and `the` (75). The spaCy `en_core_web_sm` pipeline found 66 named entities, led by ORG (31), CARDINAL (13), PERSON (7) and GPE (6). A second, toolkit-free implementation reproduced the frequency table with `str.split()` and `string.punctuation`, saving `term_frequencies_without_toolkit.csv`. Finally, a three-document TF, DF, IDF and TF-IDF pipeline was implemented manually, with `language` reaching the highest observed TF-IDF score of 0.101366 in Doc2. The CSV outputs contain 749 toolkit data rows and 750 toolkit-free data rows, corresponding to 750 and 751 lines including headers and making the effect of punctuation and tokenization rules directly visible.

## Description
The main input is `4.1_4.2_input.txt`, a 17,345-character text file with 67 lines. The notebook uses `re`, `collections.Counter`, `pandas` and `spacy` to read the file, lowercase it, extract regex word tokens, count them, sort them in descending order and write `term_frequencies.csv` with columns `Term` and `Frequency`. The regex method yields 2,446 tokens, 749 unique terms and a top-10 list headed by `a` 90, `and` 86, `the` 75, `in` 42, `to` 42 and `of` 42. A toolkit-free version reads the same file, splits on whitespace, strips leading and trailing punctuation from each token, lowercases it, counts it in a dictionary and writes `term_frequencies_without_toolkit.csv`; its top ranks agree on the first five terms but give `of` 40 instead of 42, reflecting the different treatment of punctuation and embedded word boundaries.

For named entity recognition, the notebook loads spaCy's `en_core_web_sm` model and builds a DataFrame from `ent.text`, `ent.label_` and `spacy.explain(ent.label_)`. It reports 66 entities in total and prints the first 20, including `Natural Language Processing and Artificial Int...` as ORG, `English` as LANGUAGE, `Hindi` and `Tamil` as GPE, `Assamese`, `Bengali` as NORP and several occurrences of `NLP` as ORG. The label distribution is ORG 31, CARDINAL 13, PERSON 7, GPE 6, LANGUAGE 3, NORP 3, ORDINAL 2 and DATE 1.

The TF-IDF section uses three short documents with 9, 8 and 9 tokens after lowercasing and punctuation removal. Term frequency is computed as `count / doc_len`, document frequency counts the documents containing each term, and IDF uses `math.log(num_docs / d)` over 3 documents, giving 1.098612 for terms in one document and 0.405465 for terms in two. The merged DataFrame contains 25 term-document rows; examples are `a` in Doc1 with TF 0.111111, DF 1, IDF 1.098612 and TF-IDF 0.122068, and `language` in Doc2 with TF 0.250000, DF 2, IDF 0.405465 and TF-IDF 0.101366. Per-document top-10 rankings show `a` and `field` tied at 0.122068 for Doc1, `helps`, `computers`, `understand` and `human` tied at 0.137327 for Doc2, and `machine`, `learning`, `an`, `important` and `part` tied at 0.122068 for Doc3.

## Code
### Toolkit-based frequencies and spaCy named entities
```python
import re
from collections import Counter, defaultdict
import pandas as pd
import spacy
import string

file_path = (
    "/Users/kartiken/Downloads/CLNLP-LAB/EXP-4/4.1_4.2_input.txt"
)
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

words = re.findall(r"\b\w+\b", text.lower())
word_counts = Counter(words)
df_tf = pd.DataFrame(word_counts.items(), columns=["Term", "Frequency"])
df_tf = df_tf.sort_values(by="Frequency", ascending=False).reset_index(
    drop=True
)
df_tf.to_csv("term_frequencies.csv", index=False)

top_10 = df_tf.head(10)
print("=== Top 10 Most Frequent Terms ===")
print(top_10.to_string(index=False))

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
df_ner = pd.DataFrame(
    [(ent.text, ent.label_, spacy.explain(ent.label_)) for ent in doc.ents],
    columns=["Entity", "Label", "Description"],
)
print(f"Total named entities found: {len(doc.ents)}")
df_ner.head(20)

ner_counts = Counter(ent.label_ for ent in doc.ents)
print("=== Named Entity Count by Type ===")
for label, count in ner_counts.most_common():
    print(f"{label:10} {count:4}   {spacy.explain(label)}")
```

### Toolkit-free counting and lowercase document preparation
```python
import pandas as pd
import string

file_path = "4.1_4.2_input.txt"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

split_arr = text.split()
hash_map = dict()

for x in range(len(split_arr)):
    split_arr[x] = split_arr[x].strip(string.punctuation).lower()
    hash_map[split_arr[x]] = hash_map.get(split_arr[x], 0) + 1

df_tf_2 = pd.DataFrame(hash_map.items(), columns=["Term", "Frequency"])
df_tf_2 = df_tf_2.sort_values(by="Frequency", ascending=False).reset_index(
    drop=True
)
df_tf_2.to_csv("term_frequencies_without_toolkit.csv", index=False)
df_tf_2.head(10)

doc1 = "Natural language processing is a field of artificial intelligence."
doc2 = "Natural language processing helps computers understand human language."
doc3 = "Machine learning is an important part of artificial intelligence."

doc1_arr = doc1.split()
doc2_arr = doc2.split()
doc3_arr = doc3.split()

doc1_len = len(doc1_arr)
doc2_len = len(doc2_arr)
doc3_len = len(doc3_arr)

for x in range(max(len(doc1_arr), len(doc2_arr), len(doc3_arr))):
    if x < doc1_len:
        doc1_arr[x] = doc1_arr[x].lower()
    if x < doc2_len:
        doc2_arr[x] = doc2_arr[x].lower()
    if x < doc3_len:
        doc3_arr[x] = doc3_arr[x].lower()
```

### Manual tokenization, term frequency and document frequency
```python
import pandas as pd
import string

for x in range(max(len(doc1_arr),len(doc2_arr),len(doc3_arr))):
    if x < doc1_len:
        doc1_arr[x] = doc1_arr[x].strip(string.punctuation)
    if x < doc2_len:
        doc2_arr[x] = doc2_arr[x].strip(string.punctuation)
    if x < doc3_len:
        doc3_arr[x] = doc3_arr[x].strip(string.punctuation)

doc1_tokens = []
doc2_tokens = []
doc3_tokens = []

for x in range(max(len(doc1_arr), len(doc2_arr), len(doc3_arr))):
    if x < len(doc1_arr):
        doc1_tokens.append(doc1_arr[x])
    if x < len(doc2_arr):
        doc2_tokens.append(doc2_arr[x])
    if x < len(doc3_arr):
        doc3_tokens.append(doc3_arr[x])

tokenized_docs = [doc1_tokens, doc2_tokens, doc3_tokens]
for i in range(len(tokenized_docs)):
    print(f"Doc{i + 1} ({len(tokenized_docs[i])} tokens): {tokenized_docs[i]}")

tf_rows = []

for i in range(len(tokenized_docs)):
    doc_len = len(tokenized_docs[i])
    term_counts = dict()
    for token in tokenized_docs[i]:
        term_counts[token] = term_counts.get(token, 0) + 1
    for term, count in term_counts.items():
        tf_rows.append((f"Doc{i + 1}", term, count, doc_len, count / doc_len))

df_tf_docs = pd.DataFrame(
    tf_rows, columns=["Document", "Term", "Count", "DocLength", "TF"]
)

df_counts = dict()
for tokens in tokenized_docs:
    unique_terms = set(tokens)
    for term in unique_terms:
        df_counts[term] = df_counts.get(term, 0) + 1

df_df = (
    pd.DataFrame(df_counts.items(), columns=["Term", "DF"])
    .sort_values("Term")
    .reset_index(drop=True)
)
```

### IDF, TF-IDF and per-document ranking
```python
import math

df_tf_docs = pd.DataFrame(
    tf_rows, columns=["Document", "Term", "Count", "DocLength", "TF"]
)
df_df = (
    pd.DataFrame(df_counts.items(), columns=["Term", "DF"])
    .sort_values("Term")
    .reset_index(drop=True)
)
num_docs = len(tokenized_docs)
df_df["IDF"] = df_df["DF"].apply(lambda d: math.log(num_docs / d))
df_tfidf = df_tf_docs.merge(df_df, on="Term")
df_tfidf["TF-IDF"] = df_tfidf["TF"] * df_tfidf["IDF"]
df_display = df_tfidf[["Document", "Term", "TF", "DF", "IDF", "TF-IDF"]]

for doc in df_tfidf["Document"].unique():
    top_10_doc = (
        df_tfidf[df_tfidf["Document"] == doc]
        .sort_values(by="TF-IDF", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    print(f"=== Top 10 Terms by TF-IDF for {doc} ===")
    print(top_10_doc[["Term", "TF", "DF", "IDF", "TF-IDF"]].to_string(index=False))
    print()
```

## Tables
### Corpus and token counts
| Metric | Value |
| --- | --- |
| Characters in `4.1_4.2_input.txt` | 17,345 |
| Lines | 67 |
| Regex tokens (`\b\w+\b`) | 2,446 |
| Whitespace tokens (`text.split()`) | 2,419 |
| Unique regex terms | 749 |
| Data rows in `term_frequencies.csv` | 749 |
| Data rows in `term_frequencies_without_toolkit.csv` | 750 |

### Top 10 terms from both frequency methods
| Rank | Toolkit term | Toolkit frequency | Toolkit-free term | Toolkit-free frequency |
| --- | --- | --- | --- | --- |
| 1 | a | 90 | a | 90 |
| 2 | and | 86 | and | 86 |
| 3 | the | 75 | the | 75 |
| 4 | in | 42 | in | 42 |
| 5 | to | 42 | to | 42 |
| 6 | of | 42 | can | 41 |
| 7 | can | 41 | of | 40 |
| 8 | is | 37 | is | 37 |
| 9 | word | 30 | may | 30 |
| 10 | may | 30 | word | 30 |

### spaCy named entity counts
| Label | Count | Description |
| --- | --- | --- |
| ORG | 31 | Companies, agencies, institutions, etc. |
| CARDINAL | 13 | Numerals that do not fall under another type |
| PERSON | 7 | People, including fictional |
| GPE | 6 | Countries, cities, states |
| LANGUAGE | 3 | Any named language |
| NORP | 3 | Nationalities or religious or political groups |
| ORDINAL | 2 | "first", "second", etc. |
| DATE | 1 | Absolute or relative dates or periods |
| Total | 66 | All entities found by spaCy |

### Selected TF-IDF values
| Document | Term | TF | DF | IDF | TF-IDF |
| --- | --- | --- | --- | --- | --- |
| Doc1 | natural | 0.111111 | 2 | 0.405465 | 0.045052 |
| Doc1 | a | 0.111111 | 1 | 1.098612 | 0.122068 |
| Doc2 | language | 0.250000 | 2 | 0.405465 | 0.101366 |
| Doc2 | helps | 0.125000 | 1 | 1.098612 | 0.137327 |
| Doc3 | machine | 0.111111 | 1 | 1.098612 | 0.122068 |
| Doc3 | of | 0.111111 | 2 | 0.405465 | 0.045052 |

### Top TF-IDF terms by document
| Document | Highest-scoring terms | TF-IDF |
| --- | --- | --- |
| Doc1 | a, field | 0.122068 |
| Doc2 | helps, computers, understand, human | 0.137327 |
| Doc3 | machine, learning, an, important, part | 0.122068 |

## Figures
| path | caption |
| --- | --- |
| None | |
