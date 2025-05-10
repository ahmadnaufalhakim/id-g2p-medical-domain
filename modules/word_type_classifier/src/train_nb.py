import argparse
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.utils import shuffle
import time

from utils import extract_tokens_from_file, as_minutes

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "..", "data")
MODELS_DIR = os.path.join(CURR_DIR, "..", "models", "nb")
if not os.path.exists(MODELS_DIR) :
  os.mkdir(MODELS_DIR)

if __name__ == "__main__" :
  """CORE OF THE SCRIPT"""
  parser = argparse.ArgumentParser()
  parser.add_argument("--nb_type", help="NB type (bernoulli, multinomial)", choices=["bernoulli", "multinomial"], required=True)
  parser.add_argument("--main_corpus", help="MAIN language training corpus file name", required=False)
  parser.add_argument("--foreign_corpus", help="FOREIGN language training corpus file name", required=False)
  args = parser.parse_args()
  print(args)

  nb_type = args.nb_type
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

  # Prepare the array of data to train the model on
  X = X_train
  y = y_train
  ## uncomment 2 lines below to enable training on train+val splits
  # X += X_val
  # y += y_val

  # Define all the module needed (vectorizer, normalizer, and NB model)
  n_min, n_max = 1, 3
  vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(n_min, n_max))
  normalizer = Normalizer()
  if nb_type == "bernoulli" :
    # best param: {"alpha": 0.1}
    alpha = .1
    nb_clf = BernoulliNB(alpha=alpha)
  elif nb_type == "multinomial" :
    # best param: {"alpha": 0.0}
    alpha = .0
    nb_clf = MultinomialNB(alpha=alpha)

  # Define the pipeline
  pipeline = Pipeline([
    ("tfidf", vectorizer),
    ("normalizer", normalizer),
    ("nb_clf", nb_clf)
  ])
  # Train the pipeline
  start_time = time.time()
  pipeline.fit(X, y)
  elapsed_time = time.time()-start_time
  # Save the pipeline
  joblib.dump(pipeline, os.path.join(MODELS_DIR, f"pipeline-{nb_type[0]}nb.pkl"))

  print(f"Training finished in {as_minutes(elapsed_time)}")
