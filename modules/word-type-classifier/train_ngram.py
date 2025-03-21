import argparse
from decimal import Decimal
import os
import time
import torch

from ngram_model import NGramClassifier
from utils import extract_tokens_from_file

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "data")
MODELS_DIR = os.path.join(CURR_DIR, "models", "ngrams")
if not os.path.exists(MODELS_DIR) :
  os.mkdir(MODELS_DIR)

def restricted_float(x):
  try:
    x = float(x)
  except ValueError:
    raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))
  if x < 0.0 or x > 1.0:
    raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
  return x

parser = argparse.ArgumentParser()
parser.add_argument("--n", help="n-gram's n value", type=int)
parser.add_argument("--k", help="k value for Add-k smoothing", type=restricted_float)
parser.add_argument("--train_main_corpus", help="MAIN language training corpus file name", required=False)
parser.add_argument("--train_foreign_corpus", help="FOREIGN language training corpus file name", required=False)
parser.add_argument("--val_main_corpus", help="MAIN language val corpus file name", required=False)
parser.add_argument("--val_foreign_corpus", help="FOREIGN language val corpus file name", required=False)
args = parser.parse_args()

n = args.n
k = args.k
mdl_suffix = f"-k_{f'{Decimal(k):.0e}'.replace('-', '_')}" if k and k!=0 else ""
train_main_filepath = os.path.join(DATA_DIR, "train", args.train_main_corpus) if args.train_main_corpus else os.path.join(DATA_DIR, "train", "main.csv")
train_foreign_filepath = os.path.join(DATA_DIR, "train", args.train_foreign_corpus) if args.train_foreign_corpus else os.path.join(DATA_DIR, "train", "foreign.csv")
val_main_filepath = os.path.join(DATA_DIR, "val", args.val_main_corpus) if args.val_main_corpus else os.path.join(DATA_DIR, "val", "main.csv")
val_foreign_filepath = os.path.join(DATA_DIR, "val", args.val_foreign_corpus) if args.val_foreign_corpus else os.path.join(DATA_DIR, "val", "foreign.csv")

"""CORE OF THE SCRIPT"""
# Extract tokens from filepaths
main_tokens = extract_tokens_from_file(train_main_filepath) + extract_tokens_from_file(val_main_filepath)
foreign_tokens = extract_tokens_from_file(train_foreign_filepath) + extract_tokens_from_file(val_foreign_filepath)

# Define model, then train
ngram_clf = NGramClassifier(n=n, k=k)
start_time = time.time()
ngram_clf.train(
  main_tokens=main_tokens,
  foreign_tokens=foreign_tokens
)
elapsed_time = time.time() - start_time
torch.save(obj=ngram_clf.model, f=os.path.join(MODELS_DIR, f"{ngram_clf.n}gram{mdl_suffix}.pth"))
print(f"{n}-gram model training finished in {elapsed_time} sec(s)")
