```markdown
# Project: Implementation of BM25 with Semantic Enhancement

## Description
This project implements the BM25 ranking algorithm and its modification using Word2Vec for semantic analysis. It consists of three main components:
1. **DocumentTokenizer** — Document tokenization
2. **CreateModel** — Training a Word2Vec model
3. **TestBM25** — Testing the BM25 algorithms and comparing results

Each of the files contains additional instructions.
In the repository, you can find the prepared files in the "BEIR NFCorpus" archive. It contains prepared documents. You can unzip it, create an empty database, and specify folder paths in the TestBM25 file.java and CreateModel.java and run the Training a Word2Vec model.

## Features
- Text document tokenization using BPE tokenizer
- Training a Word2Vec model on tokenized data
- Implementation of BM25 with semantic vector extension
- Comparison of two approaches (standard BM25 and semantic-enhanced BM25)
- Output of metrics: Precision, Recall, F1, MAP, nDCG

## Requirements
### Platform
- Java 17+
- PostgreSQL 12+

## Installation
### 1. Database Setup
- Create an empty database in PostgreSQL (e.g., `scopusTestTokenNF`)
- In the file `TestBM25.java`, change the parameters:
```java
static String URL_DATABASE = "jdbc:postgresql://localhost:5432/scopusTestTokenNF";
static String USER = "postgres";
static String PASSWORD = "postgres";
```

### 2. Data Preparation
- Folders with documents:
    - `\\BEIR NFCorpus\\Doc` (raw texts)
    - `\\BEIR NFCorpus\\DocToken` (tokenized documents)
- File with questions: `\\BEIR NFCorpus\\PrepQueriesBM25RAW.json`

## Usage
### Step 1: Document Tokenization

Run main in DocumentTokenizer

Results will be saved in `\\BEIR NFCorpus\\DocToken`.

### Step 2: Training Word2Vec Model

Run main in CreateModel

The model will be saved as `model300Token.bin`.

### Step 3: BM25 Testing

Run main in TestBM25

Results will be printed to the console and stored in the database.

## Configuration Parameters
In the file `TestBM25.java`, you can change the following parameters:
```java
static int maxKNN = 100; // Number of semantically similar words
static double minSim = 0.6; // Minimum similarity threshold
static double lsim = 0.8;
static int TOP_k = 10000000;
static double BM25_k = 1.7;
static double BM25_b = 0.75;
```

## Output Data
- Test results will be printed to the console in the following format:
```text
(top-10000000)
Algorithm    Precision   Recall   F1       MAP     nDCG
A           0.8564      0.7923   0.8231   0.7892  0.8643
B           0.7824      0.7215   0.7503   0.7199  0.7932
Delta(A-B)  0.074       0.0708   0.0728   0.0693  0.0711
```

## Notes
- Ensure that the file paths in the code match your file system (backslashes are used in the example)
- The test data should include documents in `.txt` format and a JSON file with questions

## License
MIT License (see the LICENSE file)

## Author
Aleksei Shrank, September 2025
```