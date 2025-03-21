import os
import re
import string

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "..", "data")
MODELS_DIR = os.path.join(CURR_DIR, "..", "models")
if not os.path.exists(MODELS_DIR) :
  os.mkdir(MODELS_DIR)

def preprocess_text(text:str) :
  """
    Return cleaned text (no dashes, digits, tabs, punctuation, non-alphabetic characters)
  """
  # Replace en em dashes with whitespace
  text = re.sub('–', ' ', text)
  text = re.sub('—', ' ', text)
  # Remove digits
  text = re.sub(r"\d", '', text)
  # Remove tabs
  text = re.sub(r"\t", '', text)
  # Remove non-alphabetic characters (except apostrophe and hyphen)
  text = re.sub(r"[^a-zA-Z\s\'-]", '', text)

  # Remove punctuation (except apostrophe and hyphen) using string.punctuation
  punctuation = f"{string.punctuation}‘’“”"
  punctuation = ''.join(char for char in punctuation if char not in "'-")
  text = text.translate(str.maketrans(punctuation, ' '*len(punctuation)))
  return text.lower().strip()

def extract_tokens_from_file(filepath:str) -> list :
  fstream = open(filepath, 'r', encoding="UTF_8").read().lower()
  clean_text = preprocess_text(fstream)
  return clean_text.split()

# SVC utils
def preprocess_train_corpus(raw_train_corpus:list, label:int, level:str="sent") :
  """
    Preprocess raw train corpus (list of strings)

    Level denotes the level of the corpus preprocessing method ("sent" or "word")
  """
  assert level == "sent" or level == "word"
  X, y = [], []
  for document in raw_train_corpus :
    if level == "sent" :
      sentence = ' '.join(preprocess_text(document).split())
      if sentence != '' and sentence not in X :
        X.append(sentence)
        y.append(label)
    else :
      tokens = preprocess_text(document).split()
      for token in tokens :
        if token != '' and token not in X :
          X.append(token)
          y.append(label)
  return X, y

def preprocess_test_corpus(raw_test_corpus:list, level:str="sent") :
  """
    Preprocess raw test corpus (list of strings)
    Each test corpus entry must be in the following syntax:
    ```
    '<class label>\\t<sentence|word>\\n'
    ```
    Level denotes the level of the corpus preprocessing method ("sent" or "word")
  """
  assert level == "sent" or level == "word"
  X, y = [], []
  for document in raw_test_corpus :
    tokens = document.split()
    if level == "sent" :
      sentence = ' '.join([preprocess_text(token) for token in tokens[1:] if preprocess_text(token) != ''])
      if sentence != '' and sentence not in X :
        X.append(sentence)
        y.append(int(tokens[0]))
    else :
      for token in tokens[1:] :
        if token != '' and token not in X :
          X.append(token)
          y.append(int(tokens[0]))
  return X, y
