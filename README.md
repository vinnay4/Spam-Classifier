---

# **Spam vs. Ham Classification (From Scratch)**

This project implements a complete end-to-end spam classification system **built entirely from scratch**, without using any machine learning libraries such as scikit-learn.
The goal is to classify messages or e-mails as **Spam (+1)** or **Ham (0)** using classic NLP and probabilistic learning techniques.

---

## **1. Objective**

Build a fully functional spam classifier using:

* Custom text preprocessing
* Custom vectorization (Bag-of-Words)
* A manually implemented Multinomial Naive Bayes classifier
* Automatic test folder scanning and prediction

---

## **2. Features**

### **Fully custom NLP pipeline**

* Lowercasing
* Tokenization
* Stopword removal
* Alphanumeric filtering
* Porter stemming

### **Hand-written Bag-of-Words vectorizer**

* Builds vocabulary from scratch
* Constructs count vectors (no external ML tools used)

### **Custom Multinomial Naive Bayes implementation**

* Manual calculation of priors, likelihoods, and log-probabilities
* Laplace smoothing
* Document classification via log-posterior scoring

### **Train/Test split**

* 80/20 randomized custom split (no ML libraries)

### **Performance evaluation from scratch**

* Accuracy
* Precision
* Confusion Matrix

### **Automated test folder classification**

* Automatically reads files named `email1.txt`, `email2.txt`, … from a folder named `test/`
* Outputs prediction: **+1 (spam)** or **0 (ham)**

### **Manual file inspection utility**

* Classify any individual text file such as `email42.txt`

---

## **3. Tech Stack**

* **Language:** Python
* **Libraries:** NLTK, NumPy, Pandas (only for tokenization and dataset handling)
* **No ML libraries used**

---

## **4. Project Workflow**

1. Load dataset (`emails.csv`)
2. Preprocess all messages
3. Build vocabulary
4. Convert messages to numerical vectors
5. Train Multinomial Naive Bayes
6. Evaluate on custom test split
7. Predict unseen emails from the `test/` directory

---

## **5. Key Learning Outcomes**

* Understanding foundational NLP techniques
* Implementing machine learning algorithms mathematically
* Handling sparse text data efficiently
* Designing reproducible training/inference pipelines
* Working without high-level ML APIs to build intuition

---

## **6. Repository Contents**

* `spam_classifier.py` – complete combined implementation
* `emails.csv` – dataset for training
* `README.md` – project overview
* `Spam Classifier Report.pdf` – detailed LaTeX report

---

## **7. How to Run**

### **Run the classifier**

```
python spam_classifier.py
```

### **Outputs include:**

* Training metrics
* Automatic predictions for all files inside `test/`

---
