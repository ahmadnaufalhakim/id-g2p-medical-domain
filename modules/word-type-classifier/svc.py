import argparse
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import utils

parser = argparse.ArgumentParser()
parser.add_argument("test_corpus", help="test corpus")
parser.add_argument('n_min', help="n-gram's min n value")
parser.add_argument('n_max', help="n-gram's max n value")
parser.add_argument("--main_train_corpus", help="MAIN language training corpus", required=False)
parser.add_argument("--foreign_train_corpus", help="FOREIGN language training corpus", required=False)
args = parser.parse_args()

# SECTION 1: Preprocess train and test corpus
print("Preprocessing train and test corpus")
raw_main_corpus = open(args.main_train_corpus if args.main_train_corpus is not None else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "train", "main.txt"), 'r', newline='', encoding="UTF_8").read().lower().split('\n')
raw_foreign_corpus = open(args.foreign_train_corpus if args.foreign_train_corpus is not None else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "train", "foreign.txt"), 'r', newline='', encoding="UTF_8").read().lower().split('\n')
raw_test_corpus = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.test_corpus), 'r', newline='', encoding="UTF_8").read().lower().split('\n')

raw_main_corpus = raw_main_corpus[:1000]
raw_foreign_corpus = raw_foreign_corpus[:1000]
raw_test_corpus = raw_test_corpus[:1000]

X_train, y_train = [], []
X_test, y_test = [], []
for x,y in zip(*utils.preprocess_train_corpus(raw_main_corpus,0,"word")) :
  if x not in X_train :
    X_train.append(x)
    y_train.append(y)
for x,y in zip(*utils.preprocess_train_corpus(raw_foreign_corpus,1,"word")) :
  if x not in X_train :
    X_train.append(x)
    y_train.append(y)
for x,y in zip(*utils.preprocess_test_corpus(raw_test_corpus, "word")) :
  if x not in X_test :
    X_test.append(x)
    y_test.append(y)
print(len(X_train), len(y_train))
print(len(X_test), len(y_test))
print("Train and test corpus preprocessed")

# SECTION 2: Train TF-IDF vectorizer
print("Training vectorizer")
n_min, n_max = int(args.n_min), int(args.n_max)
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(n_min,n_max))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("Vectorizer trained")

# SECTION 3: Train SVM Classifier
svm_classifier = SVC(kernel="linear")
print("Training SVM Classifier")
svm_classifier.fit(X_train_tfidf, y_train)
print("SVM Classifier trained")

# SECTION 4: Test SVM Classifier
print("Testing SVM Classifier")
predictions = svm_classifier.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# SECTION 5 (optional): Show misclassified terms
misclassified_tokens = []
for i, token in enumerate(X_test) :
  if predictions[i] != y_test[i] :
    misclassified_tokens.append(token)
print(len(misclassified_tokens), misclassified_tokens)