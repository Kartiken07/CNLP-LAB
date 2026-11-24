# Experiment 3: Stemming, Lemmatization and Regular Expressions

## Abstract
This experiment implements three NLP tasks with NLTK and Python's `re` module. First, the Porter Stemmer is applied to eight words, collapsing `playing`, `played` and `plays` to `play`, `studies` and `studying` to `studi`, `connected` and `connection` to `connect`, and `computers` to `comput`. Second, WordNet is loaded from `/Users/kartiken/nltk_data`, where `wordnet.synsets("dog")` returns eight synsets from `dog.n.01` to `chase.v.01`, and the default WordNet lemmatizer is applied to fourteen words. The lemmatizer maps `cats` to `cat`, `dogs` to `dog`, `children` to `child`, `mice` to `mouse` and `leaves` to `leaf`, but leaves `running`, `ran`, `better`, `went`, `ate`, `caring` and `studying` unchanged because no part-of-speech tag is supplied. Third, five regular expressions extract 2 email addresses, 3 URLs, 2 mobile numbers, 3 hashtags and 2 mentions from a short workshop announcement. The results contrast the aggressive, suffix-based behaviour of stemming with the dictionary-based behaviour of lemmatisation and demonstrate reliable structured-data extraction with regex patterns.

## Description
The notebook `exp3.1_Stemming_Lemmatization_and_Regular_Expression.ipynb` uses no external corpus: the input is an eight-word list for stemming, a fourteen-word list for lemmatisation and a seven-line inline string for regex extraction. It imports `PorterStemmer` and `WordNetLemmatizer` from `nltk.stem`, `wordnet` from `nltk.corpus` after inserting `/Users/kartiken/nltk_data` into `nltk.data.path`, and `re` for pattern extraction. Task 1 stores stems in a list named `result` and displays it, Task 2 prints one `word -> lemma` line per input word, and Task 3 prints five labelled lists.

For stemming, `ps.stem()` reduces all eight inputs: `playing`, `played` and `plays` become `play`; `studies` and `studying` both become `studi`; `connected` and `connection` both become `connect`; and `computers` becomes `comput`. The WordNet lookup for `dog` returns 8 synsets: `dog.n.01`, `frump.n.01`, `dog.n.03`, `cad.n.01`, `frank.n.02`, `pawl.n.01`, `andiron.n.01` and `chase.v.01`. This confirms that the WordNet data files are available to the lemmatizer used in the next task.

Default noun lemmatisation produces 14 printed mappings. Seven inputs change: `cats` -> `cat`, `dogs` -> `dog`, `runs` -> `run`, `studies` -> `study`, `children` -> `child`, `mice` -> `mouse` and `leaves` -> `leaf`. Seven inputs are returned unchanged: `running`, `ran`, `studying`, `better`, `went`, `ate` and `caring`, illustrating that irregular verbs and `-ing` forms are not normalised without an explicit POS argument.

The regex task defines `emails = re.findall(r'[\w.+-]+@[\w.-]+\.\w+', text)`, URL patterns for `http://`, `https://` and `www.`, Indian mobile patterns `\+91-[6-9]\d{9}` and `[6-9]\d{9}`, `hashtags` with `#\w+` and `mentions` with `\B@\w+`. The output contains `nlpworkshop@gmail.com` and `support@python.org`; `https://www.nlpworkshop.com`, `www.python.org` and `https://github.com/NLPWorkshop`; `+91-9876543210` and `9123456789`; `#NLP`, `#Python` and `#MachineLearning`; and `@NLPWorkshop` and `@PythonLearner`. All 12 extracted strings are printed without false positives from the surrounding prose.

## Code

### Porter stemming, WordNet lookup and WordNet lemmatization
```python
from nltk.stem import PorterStemmer,WordNetLemmatizer

ps= PorterStemmer()

words = ["playing", "played", "plays", "studies", "studying", "connected", "connection", "computers"]

result=[]
for i in words:
    result.append(ps.stem(i))

result

import nltk

nltk.data.path.insert(0, '/Users/kartiken/nltk_data')

from nltk.corpus import wordnet

print(wordnet.synsets("dog"))

# Experiment 3.2: Write a Python program to perform lemmatization on a given input words:
words_2 = ["cats", "dogs", "running", "runs", "ran", "studies", "studying", "better", "children", "mice", "went", "ate", "leaves", "caring"]

lemmatizer = WordNetLemmatizer()

for word in words_2:
    print(word, "->", lemmatizer.lemmatize(word))
```

### Regex extraction of emails, URLs, phones, hashtags and mentions
```python
import re

text = """Welcome to the Natural Language Processing workshop! For more information, visit
https://www.nlpworkshop.com or www.python.org. You can contact the coordinator at
nlpworkshop@gmail.com or support@python.org. For registration, call +91-9876543210 or
9123456789. Follow us on social media @NLPWorkshop and @PythonLearner. Share your
experience using #NLP, #Python, and #MachineLearning. You can also visit
https://github.com/NLPWorkshop for the latest updates."""

emails = re.findall(r'[\w.+-]+@[\w.-]+\.\w+', text)
urls = re.findall(r'http://[\w./-]+\b|https://[\w./-]+\b|www\.[\w./-]+\b', text)
mobiles = re.findall(r'\+91-[6-9]\d{9}|[6-9]\d{9}', text)
hashtags = re.findall(r'#\w+', text)
mentions = re.findall(r'\B@\w+', text)

print("Email addresses:", emails)
print("URLs:", urls)
print("Mobile numbers:", mobiles)
print("Hashtags:", hashtags)
print("Mentions:", mentions)
```

## Tables

### Porter Stemmer results for the eight input words
| Original Word | Stem |
| --- | --- |
| playing | play |
| played | play |
| plays | play |
| studies | studi |
| studying | studi |
| connected | connect |
| connection | connect |
| computers | comput |

### WordNet synsets returned for "dog"
| Query | Synset Count | Synsets |
| --- | --- | --- |
| dog | 8 | dog.n.01, frump.n.01, dog.n.03, cad.n.01, frank.n.02, pawl.n.01, andiron.n.01, chase.v.01 |

### Default WordNet lemmatizer results for the fourteen input words
| Word | Lemma |
| --- | --- |
| cats | cat |
| dogs | dog |
| running | running |
| runs | run |
| ran | ran |
| studies | study |
| studying | studying |
| better | better |
| children | child |
| mice | mouse |
| went | went |
| ate | ate |
| leaves | leaf |
| caring | caring |

### Regex extraction results from the workshop announcement
| Category | Count | Extracted Values |
| --- | --- | --- |
| Email addresses | 2 | nlpworkshop@gmail.com, support@python.org |
| URLs | 3 | https://www.nlpworkshop.com, www.python.org, https://github.com/NLPWorkshop |
| Mobile numbers | 2 | +91-9876543210, 9123456789 |
| Hashtags | 3 | #NLP, #Python, #MachineLearning |
| Mentions | 2 | @NLPWorkshop, @PythonLearner |

## Figures

| path | caption |
| --- | --- |
| None | |
