# Spam-Classifier
Spam vs. Ham Classification (From Scratch)
This project implements a complete end-to-end spam classification system from scratch, without using any machine learning libraries such as scikit-learn. The goal is to classify messages or e-mails as Spam (+1) or Ham (0) using classic NLP and probabilistic learning techniques.
1. Objective
Build a fully functional spam classifier using:

a) Custom text preprocessing
b) Custom vectorization (Bag-of-Words)
c) A manually implemented Multinomial Naive Bayes classifier
d) Automatic test folder scanning and prediction

3. Features
Fully custom NLP pipeline
Lowercasing
Tokenization
Stopword removal
Alphanumeric filtering
Porter stemming
Hand-written Bag-of-Words vectorizer
Builds vocabulary from scratch
Constructs count vectors (no external ML tools used)
Custom Multinomial Naive Bayes implementation
Manual calculation of priors, likelihoods, and log-probabilities
Laplace smoothing
Document classification via log-posterior scoring
Train/Test split without ML libraries
80/20 randomized custom split
Performance evaluation from scratch
Accuracy
Precision
Confusion Matrix
Automated test folder classification
Automatically reads files named email1.txt, email2.txt, … from a folder named test/
Outputs prediction: +1 (spam) or 0 (ham)
Manual file inspection utility
Classify any individual text file such as email42.txt
4. Tech Stack
Language: Python
Libraries used: NLTK, NumPy, Pandas (only for tokenization and dataset handling)
No machine learning libraries used
5. Project Workflow
Load dataset (emails.csv)
Preprocess all messages
Build vocabulary
Convert messages to numerical vectors
Train Multinomial Naive Bayes
Evaluate on custom test split
Predict unseen emails from the test/ directory
6. Key Learning Outcomes
Understanding foundational NLP techniques
Implementing machine learning algorithms mathematically
Handling sparse text data efficiently
Designing reproducible training/inference pipelines
Working without high-level ML APIs to build intuition
7. Repository Contents
spam_classifier.ipynb – complete combined implementation
emails.csv – dataset for training
README.md – project overview
Spam Classifier Report.pdf – detailed LaTeX report
8. How to Run
python spam_classifier.ipynb
Outputs:
Training metrics
Automatic predictions for all files in test/
