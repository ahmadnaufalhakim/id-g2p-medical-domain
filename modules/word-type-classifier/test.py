import argparse
import glob
import nltk
try :
  nltk.data.find("tokenizers/punkt")
except LookupError :
  nltk.download("punkt")
from nltk.tokenize import TweetTokenizer
import os
import sys

import utils

parser = argparse.ArgumentParser()
parser.add_argument("test_corpus", help="test corpus")
parser.add_argument("n_list", help="list of n-gram's n values used to evaluate the test corpus", type=str)
args = parser.parse_args()

# SECTION 0: Check existence of language n-grams for each n value
n_list = [int(n.strip()) for n in args.n_list.split(',')]
for n in n_list :
  fname_patterns = [
    f"*_single_{n}gram.csv",
    f"*_pre_{n}gram.csv",
    f"*_mid_{n}gram.csv",
    f"*_post_{n}gram.csv",
    f"*_all_{n}gram.csv",
  ]
  for fname_pattern in fname_patterns :
    if not glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ngrams", fname_pattern)) :
      raise Exception(f"Need to train main and foreign language n-gram for n={n} before executing this script")

# SECTION 1: Reading main and foreign lang n-grams and their probabilities from their respective *.csv files
main_ngrams_list, main_ngram_probas_list = [], []
for n in n_list :
  ngrams, ngrams_count, ngram_probas = utils.extract_lang_ngrams(n=n, fname_prefix="main")
  main_ngrams_list.append(ngrams)
  main_ngram_probas_list.append(ngram_probas)

foreign_ngrams_list, foreign_ngram_probas_list = [], []
for n in n_list :
  ngrams, ngrams_count, ngram_probas = utils.extract_lang_ngrams(n=n, fname_prefix="foreign")
  foreign_ngrams_list.append(ngrams)
  foreign_ngram_probas_list.append(ngram_probas)

# SECTION 2: Word-level language identification from test corpus using language n-gram
tokenizer = TweetTokenizer()
non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode+1), 0xfffd)
test_fstream = open(args.test_corpus, 'r', newline='', encoding="UTF_8").read().lower()
test_token_list = list()
test_frequency = {}
test_clean_text = utils.remove_digits_punctuation(test_fstream)
test_tokens = sorted(tokenizer.tokenize(test_clean_text))

# dummy_tokens = ["ini", "senyumanmu"]
# dummy_tokens.extend(test_tokens[-4:])
# test_tokens = dummy_tokens

for test_token in test_tokens :
  print(test_token, utils.calculate_lid(
    word=test_token,
    n_list=n_list,
    main_ngrams_list=main_ngrams_list, main_ngram_probas_list=main_ngram_probas_list,
    foreign_ngrams_list=foreign_ngrams_list, foreign_ngram_probas_list=foreign_ngram_probas_list
  ))
