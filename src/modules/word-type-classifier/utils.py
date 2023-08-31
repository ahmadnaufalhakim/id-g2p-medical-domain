import csv
import nltk
try :
  nltk.data.find("tokenizers/punkt")
except LookupError :
  nltk.download("punkt")
from nltk.tokenize import TweetTokenizer
import os
import re
import string

def remove_digits_punctuation(text:str) :
  """
    Return cleaned text (no hyphens, digits, tabs, punctuation, non-alphabetic characters)
  """
  # Replace hyphen with whitespace
  text = re.sub('-', ' ', text)
  text = re.sub('—', ' ', text)

  # Remove digits
  text = re.sub(r"\d", '', text)
  # Remove tabs
  text = re.sub(r"\t", '', text)
  # Remove apostrophes
  text = re.sub("'", '', text)
  # Remove non-alphabetic characters
  text = re.sub(r"[^a-zA-Z\s\']", '', text)

  # Remove punctuation (except apostrophe) using string.punctuation
  punctuation = f"{string.punctuation}‘’“”"
  text = text.translate(str.maketrans(punctuation, ' '*len(punctuation)))
  return text.strip()

def get_frequency_distribution(tokens:list) :
  """
    Returns the words frequency distribution

    Output example:
    ```
    result = {
      "word1": 1,
      "word2": 2,
      etc.
    }
  """
  result = {}
  for token in tokens :
    result[token] = result.get(token, 0) + 1
  return result

def generate_ngrams(frequency:dict, n:int) :
  """
    Returns tuple of lists of respectively `single`, `pre`, `mid`, `post`, and `all` n-grams and their occurrences

    Output structure:
    ```
    result = (
      single_ngrams, single_ngrams_count,
      pre_ngrams, pre_ngrams_count,
      mid_ngrams, mid_ngrams_count,
      post_ngrams, post_ngrams_count,
      all_ngrams, all_ngrams_count
    )
    ```
  """
  single_ngrams, single_ngrams_count = list(), list()
  pre_ngrams, pre_ngrams_count = list(), list()
  mid_ngrams, mid_ngrams_count = list(), list()
  post_ngrams, post_ngrams_count = list(), list()
  all_ngrams, all_ngrams_count = list(), list()
  for word, freq in frequency.items() :
    # Handle for single ngrams
    if len(word) == n :
      # all ngrams
      if word in all_ngrams :
        all_ngram_index = all_ngrams.index(word)
        old_all_ngram_count = all_ngrams_count[all_ngram_index]
        del all_ngrams_count[all_ngram_index]
        new_all_ngram_count = old_all_ngram_count + freq
        all_ngrams_count.insert(all_ngram_index, new_all_ngram_count)
      else :
        all_ngrams.append(word)
        all_ngrams_count.append(freq)
      single_ngrams.append(word)
      single_ngrams_count.append(freq)
    # Handle for pre, mid, and post ngrams
    elif len(word) > n :
      for i in range(len(word)-(n-1)) :
        ngram = word[i:i+n]
        # all ngrams
        if ngram in all_ngrams :
          all_ngram_index = all_ngrams.index(ngram)
          old_all_ngram_count = all_ngrams_count[all_ngram_index]
          del all_ngrams_count[all_ngram_index]
          new_all_ngram_count = old_all_ngram_count + freq
          all_ngrams_count.insert(all_ngram_index, new_all_ngram_count)
        else :
          all_ngrams.append(ngram)
          all_ngrams_count.append(freq)
        # pre ngrams
        if i == 0 :
          if ngram in pre_ngrams :
            pre_ngram_index = pre_ngrams.index(ngram)
            old_pre_ngram_count = pre_ngrams_count[pre_ngram_index]
            del pre_ngrams_count[pre_ngram_index]
            new_pre_ngram_count = old_pre_ngram_count + freq
            pre_ngrams_count.insert(pre_ngram_index, new_pre_ngram_count)
          else :
            pre_ngrams.append(ngram)
            pre_ngrams_count.append(freq)
        # mid ngrams
        elif i>0 and i<(len(word)-(n-1))-1 :
          if ngram in mid_ngrams :
            mid_ngram_index = mid_ngrams.index(ngram)
            old_mid_ngram_count = mid_ngrams_count[mid_ngram_index]
            del mid_ngrams_count[mid_ngram_index]
            new_mid_ngram_count = old_mid_ngram_count + freq
            mid_ngrams_count.insert(mid_ngram_index, new_mid_ngram_count)
          else :
            mid_ngrams.append(ngram)
            mid_ngrams_count.append(freq)
        # post ngrams
        elif i == (len(word)-(n-1))-1 :
          if ngram in post_ngrams :
            post_ngram_index = post_ngrams.index(ngram)
            old_post_ngram_count = post_ngrams_count[post_ngram_index]
            del post_ngrams_count[post_ngram_index]
            new_post_ngram_count = old_post_ngram_count + freq
            post_ngrams_count.insert(post_ngram_index, new_post_ngram_count)
          else :
            post_ngrams.append(ngram)
            post_ngrams_count.append(freq)
  return (
    single_ngrams, single_ngrams_count,
    pre_ngrams, pre_ngrams_count,
    mid_ngrams, mid_ngrams_count,
    post_ngrams, post_ngrams_count,
    all_ngrams, all_ngrams_count
  )

def generate_ngram_probabilities(ngrams:list, ngrams_count:list, all_ngrams_count:list) :
  """
    Returns all n-grams' probability value
  """
  ngram_probs = list()
  ngram_denominator = sum(all_ngrams_count)
  for ngram_index in range(len(ngrams)) :
    ngram_prob = ngrams_count[ngram_index]/ngram_denominator
    ngram_probs.append(ngram_prob)
  return ngram_probs

def write_ngrams_to_file(fname:str, ngrams:list, ngrams_count:list, ngram_probs:list) :
  """
    Writes each n-gram, n-gram count, and n-gram probabilites to a file
  """
  with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ngrams", fname), 'w') as f :
    for i in range(len(ngrams)) :
      f.write(f"{ngrams[i]},{ngrams_count[i]},{ngram_probs[i]}\n")

def read_ngrams_from_file(fname:str) :
  """
    Reads all n-gram, n-gram count, and n-gram probabilities from a file
  """
  ngrams, ngrams_count, ngram_probabilities = list(), list(), list()
  with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ngrams", fname), 'r') as f :
    csv_reader = csv.reader(f)
    for row in csv_reader :
      ngrams.append(row[0])
      ngrams_count.append(int(row[1]))
      ngram_probabilities.append(float(row[2]))
  return ngrams, ngrams_count, ngram_probabilities

# Train utils
def generate_lang_ngrams(text:str, tokenizer:TweetTokenizer, n:int, fname_prefix:str) :
  """
    Combines all defined functions to generate and write the language n-grams to external files
  """
  clean_text = remove_digits_punctuation(text)
  tokens = tokenizer.tokenize(clean_text)
  frequency = get_frequency_distribution(tokens)
  prefix = fname_prefix.strip('-_ ')

  print(f"Generating {prefix} lang {n}-grams..")
  ngrams = generate_ngrams(frequency, n)
  single_ngrams, single_ngrams_count = (list(t) for t in zip(*sorted(zip(ngrams[0], ngrams[1]))))
  pre_ngrams, pre_ngrams_count = (list(t) for t in zip(*sorted(zip(ngrams[2], ngrams[3]))))
  mid_ngrams, mid_ngrams_count = (list(t) for t in zip(*sorted(zip(ngrams[4], ngrams[5]))))
  post_ngrams, post_ngrams_count = (list(t) for t in zip(*sorted(zip(ngrams[6], ngrams[7]))))
  all_ngrams, all_ngrams_count = (list(t) for t in zip(*sorted(zip(ngrams[8], ngrams[9]))))

  print(f"Generating {prefix} lang {n}-gram probabilities..")
  single_ngram_probas = generate_ngram_probabilities(single_ngrams, single_ngrams_count, all_ngrams_count)
  pre_ngram_probas = generate_ngram_probabilities(pre_ngrams, pre_ngrams_count, all_ngrams_count)
  mid_ngram_probas = generate_ngram_probabilities(mid_ngrams, mid_ngrams_count, all_ngrams_count)
  post_ngram_probas = generate_ngram_probabilities(post_ngrams, post_ngrams_count, all_ngrams_count)
  all_ngram_probas = generate_ngram_probabilities(all_ngrams, all_ngrams_count, all_ngrams_count)

  print(f"Writing {prefix} lang {n}-grams, {n}-grams count, and {n}-gram probabilities..")
  write_ngrams_to_file(f"{prefix}_single_{n}gram.csv", single_ngrams, single_ngrams_count, single_ngram_probas)
  write_ngrams_to_file(f"{prefix}_pre_{n}gram.csv", pre_ngrams, pre_ngrams_count, pre_ngram_probas)
  write_ngrams_to_file(f"{prefix}_mid_{n}gram.csv", mid_ngrams, mid_ngrams_count, mid_ngram_probas)
  write_ngrams_to_file(f"{prefix}_post_{n}gram.csv", post_ngrams, post_ngrams_count, post_ngram_probas)
  write_ngrams_to_file(f"{prefix}_all_{n}gram.csv", all_ngrams, all_ngrams_count, all_ngram_probas)

# Test utils
def extract_lang_ngrams(n:int, fname_prefix:str) :
  """
    Returns a tuple of a language's n-grams, n-grams count, and n-gram probas
  """
  prefix = fname_prefix.strip('-_ ')
  single_ngrams, single_ngrams_count, single_ngram_probas = read_ngrams_from_file(f"{prefix}_single_{n}gram.csv")
  pre_ngrams, pre_ngrams_count, pre_ngram_probas = read_ngrams_from_file(f"{prefix}_pre_{n}gram.csv")
  mid_ngrams, mid_ngrams_count, mid_ngram_probas = read_ngrams_from_file(f"{prefix}_mid_{n}gram.csv")
  post_ngrams, post_ngrams_count, post_ngram_probas = read_ngrams_from_file(f"{prefix}_post_{n}gram.csv")
  all_ngrams, all_ngrams_count, all_ngram_probas = read_ngrams_from_file(f"{prefix}_all_{n}gram.csv")
  return (
    (single_ngrams, pre_ngrams, mid_ngrams, post_ngrams, all_ngrams),
    (single_ngrams_count, pre_ngrams_count, mid_ngrams_count, post_ngrams_count, all_ngrams_count),
    (single_ngram_probas, pre_ngram_probas, mid_ngram_probas, post_ngram_probas, all_ngram_probas)
  )

def calculate_backoff_probability(token:str, word:str, n_list:list, ngrams_list:list, ngram_probas_list:list, token_offset:int=0) :
  """
    Calculate the probability of the language of a token using the language's n-gram
    given the word and the token's index offset from the original word

    Input example:
    ```
    token = "ring"
    word = "string"
    n_list = [2, 3]
    ngrams_list = [2grams, 3grams]
    ngram_probas_list = [2gram_probas, 3gram_probas]
    token_offset = 2
    ```

    Each n-grams and n-gram probas in the list should follow this structure:
    ```
    ngrams = (single_ngrams, pre_ngrams, mid_ngrams, post_ngrams, all_ngrams)
    ngram_probas = (
      single_ngram_probas,
      pre_ngram_probas,
      mid_ngram_probas,
      post_ngram_probas,
      all_ngram_probas
    )
    ```
  """
  assert len(n_list) == len(ngrams_list) == len(ngram_probas_list)
  n = n_list[-1]
  ngram_num = len(token)-(n-1)
  evaluated_ngrams = []
  result = 0.
  # Handle for single ngrams
  if len(token) == n :
    if token in ngrams_list[-1][0] :
      result += ngram_probas_list[-1][0][ngrams_list[-1][0].index(token)]
      evaluated_ngrams.append(token+":single")
    elif len(n_list) > 1 :
      tmp = calculate_backoff_probability(token, word, n_list[:-1], ngrams_list[:-1], ngram_probas_list[:-1])
      result += tmp[0]
      evaluated_ngrams.extend(tmp[1])
  # Handle for pre, mid, and post ngrams
  elif len(token) > n :
    for i in range(token_offset, token_offset+ngram_num) :
      ngram = word[i:i+n]
      # pre ngram
      if i == 0 :
        if ngram in ngrams_list[-1][1] :
          result += ngram_probas_list[-1][1][ngrams_list[-1][1].index(ngram)]/ngram_num
          evaluated_ngrams.append(ngram+":pre")
        elif len(n_list) > 1 :
          tmp = calculate_backoff_probability(ngram, word, n_list[:-1], ngrams_list[:-1], ngram_probas_list[:-1])
          result += tmp[0]/ngram_num
          evaluated_ngrams.extend(tmp[1])
      # mid ngram
      elif i>0 and i<(len(word)-(n-1))-1 :
        if ngram in ngrams_list[-1][2] :
          result += ngram_probas_list[-1][2][ngrams_list[-1][2].index(ngram)]/ngram_num
          evaluated_ngrams.append(ngram+":mid")
        elif len(n_list) > 1 :
          tmp = calculate_backoff_probability(ngram, word, n_list[:-1], ngrams_list[:-1], ngram_probas_list[:-1], i)
          result += tmp[0]/ngram_num
          evaluated_ngrams.extend(tmp[1])
      # post ngram
      elif i == (len(word)-(n-1))-1 :
        if ngram in ngrams_list[-1][3] :
          result += ngram_probas_list[-1][3][ngrams_list[-1][3].index(ngram)]/ngram_num
          evaluated_ngrams.append(ngram+":post")
        elif len(n_list) > 1 :
          tmp = calculate_backoff_probability(ngram, word, n_list[:-1], ngrams_list[:-1], ngram_probas_list[:-1], i)
          result += tmp[0]/ngram_num
          evaluated_ngrams.extend(tmp[1])
  # Handle if n value is larger than the length of the token
  elif len(n_list) > 1 :
    tmp = calculate_backoff_probability(token, word, n_list[:-1], ngrams_list[:-1], ngram_probas_list[:-1])
    result += tmp[0]
    evaluated_ngrams.extend(tmp[1])
  return result, evaluated_ngrams

def calculate_lid(
    word:str,
    n_list:list,
    main_ngrams_list:list, main_ngram_probas_list:list,
    foreign_ngrams_list:list, foreign_ngram_probas_list:list
  ) :
  """
    Calculate the language identification (LID) index value of a word using
    both main and foreign language's n-grams and n-gram probas
  """
  lid = 0.
  for i in range(len(n_list)) :
    lid += calculate_backoff_probability(
      token=word, word=word,
      n_list=n_list[:i+1],
      ngrams_list=main_ngrams_list[:i+1], ngram_probas_list=main_ngram_probas_list[:i+1]
    )[0]
    lid -= calculate_backoff_probability(
      token=word, word=word,
      n_list=n_list[:i+1],
      ngrams_list=foreign_ngrams_list[:i+1], ngram_probas_list=foreign_ngram_probas_list[:i+1]
    )[0]
  lid /= len(n_list)
  return lid

# def calculate_lid(
#     token:str,
#     n:int,
#     ngrams:Tuple[List[str],List[str],List[str],List[str],List[str]],
#     ngram_probas:Tuple[List[float],List[float],List[float],List[float],List[float]]
#   ) :
#   """
#     Calculate language identification (LID) index value of a token using n-grams

#     Input structure:
#     ```
#     ngrams = (single_ngrams, pre_ngrams, mid_ngrams, post_ngrams, all_ngrams)
#     ngram_probas = (
#       single_ngram_probas,
#       pre_ngram_probas,
#       mid_ngram_probas,
#       post_ngram_probas,
#       all_ngram_probas
#     )
#     ```
#   """
#   lid = 0.
#   # Handle for single ngrams
#   if len(token) == n :
#     if token in ngrams[0] :
#       lid += ngram_probas[0][ngrams[0].index(token)]
#   # Handle for pre, mid, and post ngrams
#   elif len(token) > n :
#     for i in range(len(token)-(n-1)) :
#       ngram = token[i:i+n]
#       # pre ngram
#       if i == 0 :
#         if ngram in ngrams[1] :
#           lid += ngram_probas[1][ngrams[1].index(ngram)]
#       # mid ngram
#       elif i>0 and i<(len(token)-(n-1))-1 :
#         if ngram in ngrams[2] :
#           lid += ngram_probas[2][ngrams[2].index(ngram)]
#       # post ngram
#       elif i == (len(token)-(n-1))-1 :
#         if ngram in ngrams[3] :
#           lid += ngram_probas[3][ngrams[3].index(ngram)]
#   if len(token) >= n :
#     lid /= len(token)-n+1
#   return lid
