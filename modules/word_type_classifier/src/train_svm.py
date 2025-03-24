import argparse
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC, SVC
from sklearn.utils import shuffle
import time

from utils import extract_tokens_from_file, as_minutes

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "..", "data")
MODELS_DIR = os.path.join(CURR_DIR, "..", "models", "svms")
if not os.path.exists(MODELS_DIR) :
  os.mkdir(MODELS_DIR)

"""CORE OF THE SCRIPT"""
parser = argparse.ArgumentParser()
parser.add_argument("--kernel", help="SVM kernel function (linear, rbf, sigmoid)", nargs='?', const=1, default="linear")
parser.add_argument("--main_corpus", help="MAIN language training corpus file name", required=False)
parser.add_argument("--foreign_corpus", help="FOREIGN language training corpus file name", required=False)
args = parser.parse_args()
print(args)

kernel = args.kernel
assert kernel in ["linear", "rbf", "sigmoid"]
train_main_filepath = os.path.join(DATA_DIR, "train", args.main_corpus) if args.main_corpus else os.path.join(DATA_DIR, "train", "main.csv")
train_foreign_filepath = os.path.join(DATA_DIR, "train", args.foreign_corpus) if args.foreign_corpus else os.path.join(DATA_DIR, "train", "foreign.csv")
val_main_filepath = os.path.join(DATA_DIR, "val", args.main_corpus) if args.main_corpus else os.path.join(DATA_DIR, "val", "main.csv")
val_foreign_filepath = os.path.join(DATA_DIR, "val", args.foreign_corpus) if args.foreign_corpus else os.path.join(DATA_DIR, "val", "foreign.csv")

# Extract tokens from filepaths
train_main_tokens = extract_tokens_from_file(train_main_filepath)
train_foreign_tokens = extract_tokens_from_file(train_foreign_filepath)
val_main_tokens = extract_tokens_from_file(val_main_filepath)
val_foreign_tokens = extract_tokens_from_file(val_foreign_filepath)

# Construct the array of data to feed to the classifier
X_train = train_main_tokens + train_foreign_tokens
y_train = [1]*len(train_main_tokens) + [0]*len(train_foreign_tokens)
X_train, y_train = shuffle(X_train, y_train)
X_val = val_main_tokens + val_foreign_tokens
y_val = [1]*len(val_main_tokens) + [0]*len(val_foreign_tokens)
X_val, y_val = shuffle(X_val, y_val)
# Merge train and validation sets for cross-validation
X = X_train + X_val
y = y_train + y_val

# Define all the module needed (vectorizer, normalizer, and SVM model)
n_min, n_max = 1, 3
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(n_min, n_max))
normalizer = Normalizer()
if kernel=="linear" :
  C = 1.
  svm_clf = LinearSVC(class_weight="balanced", verbose=True)
else :
  if kernel=="rbf" :
    C = 1.
    gamma = 1.
    coef0 = 0.
  elif kernel=="sigmoid" :
    C = 1e2
    gamma = 1e-2
    coef0 = .5
  svm_clf = SVC(C=C, kernel=kernel, gamma=gamma, coef0=coef0, class_weight="balanced", verbose=True)

# Define the pipeline
pipeline = Pipeline([
  ("tfidf", vectorizer),
  ("normalizer", normalizer),
  ("svm_clf", svm_clf)
])
# Train the pipeline
start_time = time.time()
pipeline.fit(X, y)
elapsed_time = time.time()-start_time
# Save the pipeline
joblib.dump(pipeline, os.path.join(MODELS_DIR, f"pipeline-{kernel}.pkl"))

print(f"Training finished in {as_minutes(elapsed_time)}")
