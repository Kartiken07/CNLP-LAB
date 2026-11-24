# Experiment 2: Basic Text Preprocessing, Tokenization and Stop-Word Removal

## Abstract
This experiment implements three text-preprocessing exercises on small English corpora. The first notebook reads `2.1_text_data.txt` and applies lowercasing, punctuation removal, number removal and whitespace normalisation, counting 32 uppercase letters, 35 punctuation-or-newline removals from 32 punctuation types and 6 runs of repeated whitespace. The second notebook tokenises `2.2_tokenization_data.txt` with NLTK, spaCy and plain regular expressions, producing 110 NLTK word tokens, 9 sentences, 114 spaCy tokens with part-of-speech tags and 9 regex-split sentences. The third notebook uses a 151-word stop-word set on `2.3_clean_data.txt`, reducing 113 original tokens by 57 stop words to 56 filtered tokens. spaCy's small English model `en_core_web_sm` version 3.8.0 was installed and used for POS tagging. The results show that punctuation, digits and stop words can be removed deterministically, but that deleting newline characters as punctuation also joins neighbouring words across former line boundaries.

## Description
Three notebooks and three plain-text corpora were used: `exp-2.1_basic_text_preprocessing.ipynb` with `2.1_text_data.txt` (787 characters), `exp-2.2_basic_text_preprocessing.ipynb` with `2.2_tokenization_data.txt`, and `exp-2.3_basic_text_preprocessing.ipynb` with `2.3_clean_data.txt`. The notebooks import `pandas`, `string`, `re`, `nltk.word_tokenize`, `nltk.sent_tokenize`, `spacy` and `nltk.stem`-style utilities in the companion experiment. In the second notebook, the `!python -m spacy download en_core_web_sm` cell downloaded `en_core_web_sm==3.8.0` as a 12.8 MB wheel and reported successful installation. No trained model or external dataset beyond these local text files was required for the first and third notebooks.

The first pipeline reads the raw text, counts alphabetic uppercase characters with `isupper()` (32) and lowercases the whole string. It prints the 32 ASCII punctuation characters from `string.punctuation`, adds a newline to the set and removes every matching character, reporting 35 removals. It then deletes numeric characters with `isnumeric()`; the corpus contains 13 digits across `2026`, `1000` and `12345`. Finally, `re.findall(r'\s{2,}', content)` finds 6 runs of extra whitespace and `re.sub(r'\s+', ' ', content)` collapses them to single spaces before printing the cleaned text. Because newlines were removed as punctuation, words across former line boundaries are concatenated, for example `languagein`, `summarizationtext`, `numbersfor` and `spacesthe`.

The second notebook applies three tokenisation strategies to the same 668-character NLP paragraph. NLTK's `word_tokenize` returns 110 tokens including punctuation marks such as `(`, `)`, `,`, `.` and the hyphenated token `real-world`; `sent_tokenize` returns 9 sentences, from `Natural Language Processing (NLP) is a fascinating field of Artificial Intelligence.` to `This information can then be used to provide better responses to users.` The spaCy pipeline prints 114 token rows with POS tags, comprising 35 NOUN, 17 PUNCT, 10 ADJ, 10 ADP, 10 VERB, 9 DET, 8 PROPN, 5 AUX, 3 CCONJ, 2 PART, 2 SPACE, 1 PRON, 1 NUM and 1 ADV. A manual implementation splits the text into 9 sentences with `re.split(r'(?<=[.!?])\s+', text3)` and defines words with `re.findall(r"\b\w+(?:[-']\w+)*\b|\S", text3)`; because both helper functions close over `text3` instead of using their `text` argument, every displayed per-sentence token list is the full 110-token document.

The third notebook reads the clean corpus, prints the original text and tokenises with `re.findall(r"\b\w+(?:[-']\w+)*\b", text)`, obtaining 113 tokens. It checks each lowercased token against a 151-entry `stop_words` set, finds 57 stop words including `is`, `an`, `of`, `the`, `and`, `to` and `can`, and keeps 56 content tokens beginning with `Natural`, `Language`, `Processing`, `important`, `field` and `Artificial`. The printed counts satisfy 113 - 57 = 56, and the experiment demonstrates tokenisation and stop-word filtering without external NLP libraries.

## Code

### Lowercasing, punctuation, number and whitespace cleaning (2.1)
```python
import string
import re

with open('/Users/kartiken/Downloads/CLNLP-LAB/EXP-2/2.1_text_data.txt', 'r') as file:
    content = file.read()

upper_count = 0
for i in content:
    if i.isalpha():
        if i.isupper():
            upper_count += 1
print(upper_count)

content = content.lower()

punctuation_set = set(string.punctuation)
print(punctuation_set)

punch_count = 0
punctuation_set.add("\n")
for i in content:
    if i in punctuation_set:
        content = content.replace(i, '')
        punch_count += 1
print(punch_count)

num_count = 0
for i in content:
    if i.isnumeric():
        content = content.replace(i, '')
        num_count += 1

extra_space_count = len(re.findall(r'\s{2,}', content))
content = re.sub(r'\s+', ' ', content)
print(f"Extra whitespace count: {extra_space_count}")
print(content)
```

### NLTK tokenisation and spaCy POS tagging (2.2)
```python
import pandas as pd
import nltk
from nltk import word_tokenize, sent_tokenize

with open("/Users/kartiken/Downloads/CLNLP-LAB/EXP-2/2.2_tokenization_data.txt", "r") as f:
    text = f.read()
text
print(word_tokenize(text))
print(sent_tokenize(text))

with open("/Users/kartiken/Downloads/CLNLP-LAB/EXP-2/2.2_tokenization_data.txt", "r") as f:
    text2 = f.read()
text2

import spacy
nlp = spacy.load("en_core_web_sm")
text = "SpaCy is a powerful library for Natural Language Processing."
doc = nlp(text2)
print("Token\t\tPOS Tag")
print("-----------------------")
for token in doc:
    print(f"{token.text}\t\t{token.pos_}")
```

### Regular-expression tokenisation without NLTK or spaCy (2.2)
```python
import re

with open("/Users/kartiken/Downloads/CLNLP-LAB/EXP-2/2.2_tokenization_data.txt", "r") as f:
    text3 = f.read()

def sentence_tokenize(text):
    sentences = re.split(r'(?<=[.!?])\s+', text3)
    return [s.strip() for s in sentences if s.strip()]

def word_tokenize(text):
    tokens = re.findall(r"\b\w+(?:[-']\w+)*\b|\S", text3)
    return tokens

sentences = sentence_tokenize(text3)
print("Sentence Tokenization:")
for i, sent in enumerate(sentences, 1):
    print(f"  {i}. {sent}")

print("\nWord Tokenization (all sentences):")
for sent in sentences:
    print(f"  {sent}")
    print(f"  Tokens: {word_tokenize(sent)}\n")
```

### Tokenisation and stop-word removal (2.3)
```python
import re

with open("/Users/kartiken/Downloads/CLNLP-LAB/EXP-2/2.3_clean_data.txt", "r") as f:
    text = f.read()

print("Original Text:")
print(text)

stop_words = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
    'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
    'against', 'between', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
    'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn',
    'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn',
    'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
}

def word_tokenize(text):
    return re.findall(r"\b\w+(?:[-']\w+)*\b", text)

original_tokens = word_tokenize(text)
found_stop_words = [t for t in original_tokens if t.lower() in stop_words]
filtered_tokens = [t for t in original_tokens if t.lower() not in stop_words]

print(f"Total original tokens: {len(original_tokens)}")
print(found_stop_words)
print(f"\nTotal stop words: {len(found_stop_words)}")
print(filtered_tokens)
print(f"\nTotal filtered tokens: {len(filtered_tokens)}")
```

## Tables

### Cleaning counts from 2.1_text_data.txt
| Metric | Value |
| --- | --- |
| Characters in source file | 787 |
| Uppercase alphabetic characters found | 32 |
| Punctuation types printed from string.punctuation | 32 |
| Punctuation and newline characters removed | 35 |
| Numeric characters removed | 13 |
| Runs of two or more whitespace characters | 6 |

### Tokenisation counts from 2.2_tokenization_data.txt
| Method | Word Tokens | Sentences |
| --- | --- | --- |
| NLTK word_tokenize / sent_tokenize | 110 | 9 |
| spaCy en_core_web_sm 3.8.0 | 114 | None |
| Regex split and re.findall (whole text) | 110 | 9 |

### spaCy part-of-speech distribution printed in 2.2
| POS Tag | Count |
| --- | --- |
| NOUN | 35 |
| PUNCT | 17 |
| ADJ | 10 |
| ADP | 10 |
| VERB | 10 |
| DET | 9 |
| PROPN | 8 |
| AUX | 5 |
| CCONJ | 3 |
| PART | 2 |
| SPACE | 2 |
| PRON | 1 |
| NUM | 1 |
| ADV | 1 |

### Stop-word removal from 2.3_clean_data.txt
| Metric | Value |
| --- | --- |
| Stop words in the defined set | 151 |
| Original tokens | 113 |
| Stop words found | 57 |
| Filtered tokens | 56 |
| First six filtered tokens | Natural, Language, Processing, important, field, Artificial |

## Figures

| path | caption |
| --- | --- |
| None | |
