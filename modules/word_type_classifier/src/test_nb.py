import argparse
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils import shuffle

from utils import extract_tokens_from_file

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "..", "data")
MODELS_DIR = os.path.join(CURR_DIR, "..", "models", "nb")
if not os.path.exists(MODELS_DIR) :
  raise

if __name__ == "__main__" :
  """CORE OF THE SCRIPT"""
  parser = argparse.ArgumentParser()
  parser.add_argument("--nb_type", help="NB type (bernoulli, multinomial)", choices=["bernoulli", "multinomial"], required=True)
  parser.add_argument("--main_corpus", help="MAIN language test corpus file name", required=False)
  parser.add_argument("--foreign_corpus", help="FOREIGN language test corpus file name", required=False)
  args = parser.parse_args()
  print(args)

  nb_type = args.nb_type
  test_main_filepath = os.path.join(DATA_DIR, "test", args.main_corpus) if args.main_corpus else os.path.join(DATA_DIR, "test", "main.csv")
  test_foreign_filepath = os.path.join(DATA_DIR, "test", args.foreign_corpus) if args.foreign_corpus else os.path.join(DATA_DIR, "test", "foreign.csv")

  # Extract tokens from filepaths
  test_main_tokens = extract_tokens_from_file(test_main_filepath)
  test_foreign_tokens = extract_tokens_from_file(test_foreign_filepath)

  # Construct the array of data to feed to the classifier
  X_test = test_main_tokens + test_foreign_tokens
  y_test = [1]*len(test_main_tokens) + [0]*len(test_foreign_tokens)
  X_test, y_test = shuffle(X_test, y_test)

  # Load the pipeline
  pipeline = joblib.load(os.path.join(MODELS_DIR, f"pipeline-{nb_type[0]}nb.pkl"))

  # Predict all test tokens
  y_pred = pipeline.predict(X_test)
  print(classification_report(y_test, y_pred))
  print(accuracy_score(y_test, y_pred))
  print(f1_score(y_test, y_pred))
