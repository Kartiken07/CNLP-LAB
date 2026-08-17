EXP 1: Employee Data Analysis & Visualization
# Experiment 1: Employee Data Analysis & Visualization

## Objective
To perform exploratory data analysis and visualization on an employee dataset
using Python's Pandas and Matplotlib libraries.

## Dataset
- File: employee_information_100.csv
- Records: 100 synthetic employee entries
- Columns: Employee_ID, Name, Department, Age, Gender, Salary, Experience_Years
- Departments: HR, IT, Finance, Marketing, Operations, Sales

## Tools & Libraries
- Python 3.13.2
- Pandas (data manipulation)
- Matplotlib (visualization)

## Sub-Experiments

### 1.1 Average Salary Per Department
Groups employees by department and calculates mean salary.
Displays results using a bar chart.

### 1.2 Number of Employees Per Department
Counts employees in each department.
Displays results using a bar chart.

### 1.3 Gender Distribution
Calculates male vs female percentage across the company.
Displays results using a pie chart.

### 1.4 Salary Distribution
Plots a histogram showing the spread of salaries across all employees.

### 1.5 Experience vs Salary Correlation
Creates a scatter plot to visualize the relationship between
years of experience and salary.

### 1.6 Top 10 Highest Salaries
Sorts employees by salary in descending order and displays
the top 10 earners using a bar chart.

### 1.7 Highest Salary Per Department
Finds the maximum salary in each department.
Displays results using a bar chart.

### 1.8 Employees Above Average Salary
Filters and lists employees whose salary is above the
overall company average.

### 1.9 Average Experience Per Department
Calculates the mean years of experience for each department.

### 1.10 Age Distribution
Plots a histogram showing the distribution of employee ages.

## Key Concepts
- Pandas groupby operations
- Data aggregation (mean, count, max, min)
- Bar charts, pie charts, histograms, scatter plots
- Boolean filtering for data subset selection
EXP 2.1: Basic Text Cleaning
# Experiment 2.1: Basic Text Cleaning

## Objective
To perform manual text preprocessing by cleaning raw text data
through a step-by-step pipeline.

## Dataset
- File: 2.1_text_data.txt
- Content: A passage about NLP with intentional noise
  (uppercase letters, punctuation, numbers, extra whitespace)

## Tools & Libraries
- Python 3.13.2
- Pandas
- string (for punctuation detection)
- re (regular expressions for whitespace normalization)

## Cleaning Pipeline

### Step 1: Lowercase Conversion
- Counts uppercase characters in the original text
- Converts entire text to lowercase
- Purpose: Ensures uniformity so "Hello" and "hello" are treated as the same token

### Step 2: Punctuation Removal
- Identifies all punctuation characters using string.punctuation
- Removes each punctuation mark from the text
- Counts distinct punctuation types found
- string.punctuation contains: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

### Step 3: Number Removal
- Scans text for numeric digits using isnumeric() check
- Removes all numbers from the text
- Counts digits found and removed

### Step 4: Extra Whitespace Removal
- Uses regex pattern \s+ to match one or more whitespace characters
- Replaces them with a single space using re.sub()
- Collapses multiple spaces, tabs, and newlines into single spaces

### Step 5: Final Output
- Displays the cleaned text after all preprocessing steps
- Prints counts of removed items at each step for reference

## Key Concepts
- Character-level text processing
- string.punctuation for punctuation detection
- Regular expressions for whitespace normalization
- Manual text cleaning pipeline
- Importance of preprocessing in NLP
EXP 2.2: Text Tokenization
# Experiment 2.2: Text Tokenization

## Objective
To perform text tokenization using three different approaches:
NLTK library, spaCy library, and manual regex-based method.

## Dataset
- File: 2.2_tokenization_data.txt
- Content: A passage about NLP for tokenization practice

## Tools & Libraries
- Python 3.13.2
- NLTK (Natural Language Toolkit)
- spaCy (with en_core_web_sm model)
- re (regular expressions)
- Pandas

## Approaches

### Approach 1: Using NLTK
- Word Tokenization: Uses nltk.word_tokenize() to split text into individual words
- Sentence Tokenization: Uses nltk.sent_tokenize() to split text into sentences
- NLTK is a widely used library for educational and research NLP tasks
- Slower than spaCy but excellent for learning

### Approach 2: Using spaCy
- Loads the en_core_web_sm English language model
- Processes text through spaCy's NLP pipeline
- Tokenizes text and assigns Part-of-Speech (POS) tags to each token
- POS Tags used:
  - PROPN: Proper Noun (e.g., Alice, India)
  - NOUN: Common Noun (e.g., dog, car)
  - VERB: Verb (e.g., run, eat)
  - ADJ: Adjective (e.g., big, red)
  - DET: Determiner (e.g., the, a)
- spaCy is optimized for production use with fast C-based processing

### Approach 3: Without Libraries (Manual)
- Uses regex for sentence tokenization:
  - re.split() to split on sentence-ending punctuation followed by whitespace
- Uses regex for word tokenization:
  - re.findall() with word boundary pattern to extract individual words
- Useful when external libraries are not available
- Demonstrates understanding of regex patterns

## Comparison
| Feature       | NLTK           | spaCy              | Manual Regex |
|---------------|----------------|--------------------|--------------|
| Speed         | Slow           | Fast (C-optimized)| Fast         |
| Accuracy      | Good           | Best               | Basic        |
| POS Tagging   | Available      | Built-in, accurate | Not available|
| Ease of Use   | Easy           | Easy               | Requires regex knowledge |
| Use Case      | Learning       | Production         | Simple/custom tasks |

## Key Concepts
- Sentence tokenization vs word tokenization
- Part-of-Speech (POS) tagging
- NLP library comparison (NLTK vs spaCy)
- Regex-based tokenization
- Token as the basic unit of NLP processing
EXP 2.3: Stop Word Removal
# Experiment 2.3: Stop Word Removal

## Objective
To identify and remove stop words from tokenized text using a
manual approach without relying on external NLP libraries.

## Dataset
- File: 2.3_clean_data.txt
- Content: Cleaned text ready for stop word removal

## Tools & Libraries
- Python 3.13.2
- re (regular expressions for tokenization)

## Process

### Step 1: Tokenization
- Uses regex pattern r"\b\w+(?:[-']\w+)*\b" to extract word tokens
- Pattern handles contractions like "don't" and hyphenated words
- Extracts 113 total tokens from the input text

### Step 2: Stop Word Identification
- Defines a comprehensive manual stop word set containing 126 common English words
- Includes: the, is, a, an, in, of, and, to, it, that, this, for, on, with, etc.
- Searches the tokenized text for all matching stop words
- Found 57 stop words in the text (approximately 50% of all tokens)

### Step 3: Stop Word Removal
- Filters out all stop words from the token list
- Uses case-insensitive matching (token.lower() check) to catch capitalized stop words
- Retains only content-bearing, meaningful words
- Result: 56 meaningful tokens remaining after filtering

### Step 4: Statistics
- Original tokens: 113
- Stop words found and removed: 57
- Remaining meaningful tokens: 56
- Stop word ratio: ~50% (normal for English text)

## Why Stop Words Are Removed
- They appear in almost every document and do not help distinguish between texts
- Reduces vocabulary size and processing time
- Improves model performance by focusing on semantically meaningful words
- Common in search engines, text classification, and information retrieval

## Key Concepts
- Stop words in NLP
- Manual stop word list construction
- Case-insensitive filtering
- Token count reduction (40-60% is normal for English)
- Content words vs function words
- Importance of stop word removal in NLP pipelines
