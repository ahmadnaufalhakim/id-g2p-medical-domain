import argparse
import os
from pprint import pprint

from ngram_model import NGramClassifier
from utils import extract_tokens_from_file

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "data")
MODELS_DIR = os.path.join(CURR_DIR, "models", "ngrams")
if not os.path.exists(MODELS_DIR) :
  raise

parser = argparse.ArgumentParser()
parser.add_argument("--n", help="n-gram's n value", type=int)
parser.add_argument("--main_test_corpus", help="MAIN language test corpus file name", required=False)
parser.add_argument("--foreign_test_corpus", help="FOREIGN language test corpus file name", required=False)
args = parser.parse_args()

n = args.n
main_filepath = os.path.join(DATA_DIR, "test", args.main_test_corpus) if args.main_test_corpus else os.path.join(DATA_DIR, "test", "main.csv")
foreign_filepath = os.path.join(DATA_DIR, "test", args.foreign_test_corpus) if args.foreign_test_corpus else os.path.join(DATA_DIR, "test", "foreign.csv")

"""CORE OF THE SCRIPT"""
main_tokens = extract_tokens_from_file(main_filepath)
foreign_tokens = extract_tokens_from_file(foreign_filepath)

ngram_clf = NGramClassifier(n=n)
ngram_clf.load()
evaluation_stats = ngram_clf.evaluate(
  main_words=main_tokens,
  foreign_words=foreign_tokens
)
pprint(evaluation_stats)
