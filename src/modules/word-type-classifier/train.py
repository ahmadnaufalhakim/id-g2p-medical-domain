import argparse
import nltk
try :
  nltk.data.find("tokenizers/punkt")
except LookupError :
  nltk.download("punkt")
from nltk.tokenize import TweetTokenizer
import os
import time

import utils

parser = argparse.ArgumentParser()
parser.add_argument('n', help="n-gram's n value")
parser.add_argument("--main_train_corpus", help="MAIN language training corpus", required=False)
parser.add_argument("--foreign_train_corpus", help="FOREIGN language training corpus", required=False)
args = parser.parse_args()

tokenizer = TweetTokenizer()

start_time = time.time()
# SECTION 1: N-gram generation for main lang
main_fstream = open(args.main_train_corpus if args.main_train_corpus is not None else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "train", "main.txt"), 'r', newline='', encoding="UTF_8").read().lower()
main_token_list = list()
main_frequency = {}
utils.generate_lang_ngrams(text=main_fstream, tokenizer=tokenizer, n=int(args.n), fname_prefix="main")

# SECTION 2: N-gram generation for foreign lang
foreign_fstream = open(args.foreign_train_corpus if args.foreign_train_corpus is not None else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "train", "foreign.txt"), 'r', newline='', encoding="UTF_8").read().lower()
foreign_token_list = list()
foreign_frequency = {}
utils.generate_lang_ngrams(text=foreign_fstream, tokenizer=tokenizer, n=int(args.n), fname_prefix="foreign")
elapsed_time = time.time() - start_time

print(f"Training finished in {elapsed_time} sec(s)")
