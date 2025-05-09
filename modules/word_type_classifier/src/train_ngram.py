import argparse
from decimal import Decimal
import os
import time
import torch

from ngram_model import NGramClassifier
from utils import extract_tokens_from_file

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "..", "data")
MODELS_DIR = os.path.join(CURR_DIR, "..", "models", "ngram")
if not os.path.exists(MODELS_DIR) :
  os.mkdir(MODELS_DIR)

def restricted_float(x) :
  try:
    x = float(x)
  except ValueError :
    raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))
  if x < 0.0 or x > 1.0 :
    raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
  return x

if __name__ == "__main__" :
  """CORE OF THE SCRIPT"""
  parser = argparse.ArgumentParser()
  parser.add_argument("--n", help="n-gram's n value", type=int)
  parser.add_argument("--k", help="k value for Add-k smoothing", type=restricted_float, nargs='?', const=1, default=0.)
  parser.add_argument("--main_corpus", help="MAIN language training corpus file name", required=False)
  parser.add_argument("--foreign_corpus", help="FOREIGN language training corpus file name", required=False)
  args = parser.parse_args()

  n = args.n
  k = args.k
  mdl_suffix = f"-k_{f'{Decimal(k):.0e}'.replace('-', '_')}" if k and k!=0 else ""
  train_main_filepath = os.path.join(DATA_DIR, "train", args.main_corpus) if args.main_corpus else os.path.join(DATA_DIR, "train", "main.csv")
  train_foreign_filepath = os.path.join(DATA_DIR, "train", args.foreign_corpus) if args.foreign_corpus else os.path.join(DATA_DIR, "train", "foreign.csv")
  val_main_filepath = os.path.join(DATA_DIR, "val", args.main_corpus) if args.main_corpus else os.path.join(DATA_DIR, "val", "main.csv")
  val_foreign_filepath = os.path.join(DATA_DIR, "val", args.foreign_corpus) if args.foreign_corpus else os.path.join(DATA_DIR, "val", "foreign.csv")

  # Extract tokens from filepaths
  train_main_tokens = extract_tokens_from_file(train_main_filepath)
  train_foreign_tokens = extract_tokens_from_file(train_foreign_filepath)
  val_main_tokens = extract_tokens_from_file(val_main_filepath)
  val_foreign_tokens = extract_tokens_from_file(val_foreign_filepath)

  # Prepare tokens to train the model on
  main_tokens = train_main_tokens
  foreign_tokens = train_foreign_tokens
  ## uncomment 2 lines below to enable training on train+val splits
  # main_tokens += val_main_tokens
  # foreign_tokens += val_foreign_tokens

  # Define model, then train
  ngram_clf = NGramClassifier(n=n, k=k)
  start_time = time.time()
  ngram_clf.train(
    main_tokens=main_tokens,
    foreign_tokens=foreign_tokens
  )
  elapsed_time = time.time() - start_time
  # Save the model
  torch.save(obj=ngram_clf.model, f=os.path.join(MODELS_DIR, f"{ngram_clf.n}gram{mdl_suffix}.pth"))

  print(f"{n}-gram model training finished in {elapsed_time} sec(s)")
