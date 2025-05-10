import argparse
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.preprocessing import Normalizer
from sklearn.utils import shuffle
import time

from utils import extract_tokens_from_file, plot_nb_score, as_minutes

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "..", "data")
OUTPUT_DIR = os.path.join(CURR_DIR, "..", "output", "nb")
if not os.path.exists(OUTPUT_DIR) :
  os.mkdir(OUTPUT_DIR)

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
  # Merge train and validation sets for cross-validation
  X = X_train + X_val
  y = y_train + y_val

  # Training vectorizer
  n_min, n_max = 1, 3
  vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(n_min, n_max))
  X_tfidf = vectorizer.fit_transform(X)
  # Training normalizer
  normalizer = Normalizer()
  X_tfidf = normalizer.fit_transform(X_tfidf)

  # Training NB model
  ## Define parameter grid
  param_grid = {
    "alpha": [.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
  }
  nb_clf = BernoulliNB() if nb_type == "bernoulli" else MultinomialNB()
  ## Perform grid search
  grid_search = GridSearchCV(
    estimator=nb_clf,
    param_grid=param_grid,
    scoring="f1",
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=23522026),
    n_jobs=-1,
    verbose=3
  )
  start_time = time.time()
  grid_search.fit(X_tfidf, y)
  elapsed_time = time.time()-start_time

  print(f"Training finished in {as_minutes(elapsed_time)}")
  print(f"Best parameters: {grid_search.best_params_}")
  print(f"Best cross-validation score: {grid_search.best_score_}")
  plot_nb_score(
    grid_search=grid_search,
    nb_type=nb_type,
    metric="f1"
  )
