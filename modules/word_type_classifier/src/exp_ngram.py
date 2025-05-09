import argparse
import os
import time

from ngram_model import NGramClassifier
from utils import extract_tokens_from_file, plot_ngram_score

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "..", "data")
OUTPUT_DIR = os.path.join(CURR_DIR, "..", "output", "ngram")
if not os.path.exists(OUTPUT_DIR) :
  os.mkdir(OUTPUT_DIR)

if __name__ == "__main__" :
  """CORE OF THE SCRIPT"""
  parser = argparse.ArgumentParser()
  parser.add_argument("--n", help="n-gram's n value", type=int)
  parser.add_argument("--main_corpus", help="MAIN language training corpus file name", required=False)
  parser.add_argument("--foreign_corpus", help="FOREIGN language training corpus file name", required=False)
  args = parser.parse_args()

  n = args.n
  train_main_filepath = os.path.join(DATA_DIR, "train", args.main_corpus) if args.main_corpus else os.path.join(DATA_DIR, "train", "main.csv")
  train_foreign_filepath = os.path.join(DATA_DIR, "train", args.foreign_corpus) if args.foreign_corpus else os.path.join(DATA_DIR, "train", "foreign.csv")
  val_main_filepath = os.path.join(DATA_DIR, "val", args.main_corpus) if args.main_corpus else os.path.join(DATA_DIR, "val", "main.csv")
  val_foreign_filepath = os.path.join(DATA_DIR, "val", args.foreign_corpus) if args.foreign_corpus else os.path.join(DATA_DIR, "val", "foreign.csv")
  k_list = [.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]

  # Extract tokens from filepaths
  train_main_tokens = extract_tokens_from_file(train_main_filepath)
  train_foreign_tokens = extract_tokens_from_file(train_foreign_filepath)
  val_main_tokens = extract_tokens_from_file(val_main_filepath)
  val_foreign_tokens = extract_tokens_from_file(val_foreign_filepath)

  accs = [[] for _ in range(n)]
  f1_stats = {i: {"main": [], "foreign": [], "macro": []} for i in range(n)}
  for i in range(1, n+1) :
    for k in k_list :
      ngram_clf = NGramClassifier(n=i, k=k)
      start_time = time.time()
      ngram_clf.train(
        main_tokens=train_main_tokens,
        foreign_tokens=train_foreign_tokens
      )
      elapsed_time = time.time() - start_time
      print(f"{i}-gram model (with k = {k}) training finished in {elapsed_time} sec(s)")
      evaluation_stats = ngram_clf.evaluate(
        main_words=val_main_tokens,
        foreign_words=val_foreign_tokens
      )
      accs[i-1].append(round(evaluation_stats["accuracy"]*100, 6))
      f1_stats[i-1]["main"].append(round(evaluation_stats["f1"]["main"], 6))
      f1_stats[i-1]["foreign"].append(round(evaluation_stats["f1"]["foreign"], 6))
      f1_stats[i-1]["macro"].append(round(evaluation_stats["f1"]["macro"], 6))
    plot_ngram_score(title="f1 score", n=i, main=f1_stats[i-1]["main"], foreign=f1_stats[i-1]["foreign"], macro=f1_stats[i-1]["macro"])
    plot_ngram_score(title="accuracy", n=i, accuracy=accs[i-1])
    print()

  # display stats (accs and f1-scores)
  print("accs")
  for i in range(n) :
    print(f"{i+1}gram\t{[str(acc).replace('.', ',') for acc in accs[i]]}")
  print("f1 score")
  for i in range(n) :
    print(f"{i+1}gram\t{[str(f1).replace('.', ',') for f1 in f1_stats[i]['macro']]}")
