from decimal import Decimal
from math import log, exp
import os
import torch
from typing import List, Tuple

from utils import preprocess_text

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURR_DIR, "..", "models", "ngrams")
if not os.path.exists(MODELS_DIR) :
  os.mkdir(MODELS_DIR)

class NGramClassifier :
  def __init__(self, n:int, k:float=None) :
    assert n>0
    self.n = n
    assert (k is None) or (k is not None and 0<=k<=1)
    self.k = k if k is not None else 0.
    self.mdl_suffix = f"-k_{f'{Decimal(k):.0e}'.replace('-', '_')}" if k and k!=0 else ""
    self.model = {
      'k': k if k is not None else 0.,
      "main": {i: {"single": {}, "pre": {}, "mid": {}, "post": {}, "all": {}, 'N': 0, 'V': 0, 'W': 0} for i in range(1, self.n+1)},
      "foreign": {i: {"single": {}, "pre": {}, "mid": {}, "post": {}, "all": {}, 'N': 0, 'V': 0, 'W': 0} for i in range(1, self.n+1)},
    }
    self.epsilon = 1e-12

  def load(self, mdl_path:str = None) -> None :
    if mdl_path is None :
      mdl_path = os.path.join(MODELS_DIR, f"{self.n}gram{self.mdl_suffix}.pth")
    self.model = torch.load(f=mdl_path, weights_only=True)

  def train(self, main_tokens:list, foreign_tokens:list) -> None :
    assert main_tokens and foreign_tokens
    # Generate n-grams and n-grams count (both main and foreign)
    main_word_to_freq = self.__get_frequency_distribution(main_tokens)
    foreign_word_to_freq = self.__get_frequency_distribution(foreign_tokens)
    # Train n-gram model
    for n in range(1, self.n+1) :
      # If lower n-gram order model exists, load the model
      if os.path.exists(os.path.join(MODELS_DIR, f"{n}gram{self.mdl_suffix}.pth")) :
        prev_model = torch.load(f=os.path.join(MODELS_DIR, f"{n}gram{self.mdl_suffix}.pth"), weights_only=True)
        self.model["main"][n] = prev_model["main"][n]
        self.model["foreign"][n] = prev_model["foreign"][n]
        del prev_model
        continue
      main_ngrams, main_ngrams_count = self.__generate_ngrams(main_word_to_freq, n)
      foreign_ngrams, foreign_ngrams_count = self.__generate_ngrams(foreign_word_to_freq, n)
      # Convert n-grams and their counts to model object, then save the n-gram model
      self.__save_ngrams_to_model(n, main_ngrams, main_ngrams_count, foreign_ngrams, foreign_ngrams_count)

  def evaluate(self, main_words:str, foreign_words:str) -> dict :
    """
      Evaluate model performance on main and foreign language data.
      Returns metrics: accuracy, precision, recall, F1 (for both classes), and neutral rate.
    """
    assert main_words and foreign_words
    # Prepare references (1:main, 0:foreign)
    refs = [1]*len(main_words) + [0]*len(foreign_words)
    all_words = main_words + foreign_words
    # Get model predictions
    hyps = [self.predict(word)[0][1] for word in all_words]

    # Initialize confusion matrix (rows: actual, columns: predicted)
    confusion_matrix = {
      "main": {"main": 0, "foreign": 0, "neutral": 0},
      "foreign": {"main": 0, "foreign": 0, "neutral": 0}
    }
    # Populate confusion matrix
    for ref, hyp in zip(refs, hyps) :
      ref_class = "main" if ref==1 else "foreign"
      hyp_class = "main" if hyp==1 else ("foreign" if hyp==0 else "neutral")
      confusion_matrix[ref_class][hyp_class] += 1

    # Calculate TP, FP, FN for each lang
    tp_main = confusion_matrix["main"]["main"]
    fp_main = confusion_matrix["foreign"]["main"]
    fn_main = confusion_matrix["main"]["foreign"] + confusion_matrix["main"]["neutral"]
    tp_foreign = confusion_matrix["foreign"]["foreign"]
    fp_foreign = confusion_matrix["main"]["foreign"]
    fn_foreign = confusion_matrix["foreign"]["main"] + confusion_matrix["foreign"]["neutral"]

    # Calculate evaluation metrics 
    ## accuracy and neutral rate
    accuracy = (tp_main + tp_foreign) / len(refs)
    neutral_rate = (confusion_matrix["main"]["neutral"] + confusion_matrix["foreign"]["neutral"]) / len(refs)
    ## precision
    precision_main = tp_main / (tp_main + fp_main) if (tp_main + fp_main) > 0 else 0
    precision_foreign = tp_foreign / (tp_foreign + fp_foreign) if (tp_foreign + fp_foreign) > 0 else 0
    precision_macro = (precision_main + precision_foreign) / 2
    precision_weighted = ((precision_main*self.model["main"][self.n]['W']) + (precision_foreign*self.model["foreign"][self.n]['W'])) / (self.model["main"][self.n]['W']+self.model["foreign"][self.n]['W'])
    ## recall
    recall_main = tp_main / (tp_main + fn_main) if (tp_main + fn_main) > 0 else 0
    recall_foreign = tp_foreign / (tp_foreign + fn_foreign) if (tp_foreign + fn_foreign) > 0 else 0
    recall_macro = (recall_main + recall_foreign) / 2
    recall_weighted = ((recall_main*self.model["main"][self.n]['W']) + (recall_foreign*self.model["foreign"][self.n]['W'])) / (self.model["main"][self.n]['W']+self.model["foreign"][self.n]['W'])
    ## f1-score
    f1_main = 2 * (precision_main * recall_main) / (precision_main + recall_main) if (precision_main + recall_main) > 0 else 0
    f1_foreign = 2 * (precision_foreign * recall_foreign) / (precision_foreign + recall_foreign) if (precision_foreign + recall_foreign) > 0 else 0
    f1_macro = (f1_main + f1_foreign) / 2
    f1_weighted = ((f1_main*self.model["main"][self.n]['W']) + (f1_foreign*self.model["foreign"][self.n]['W'])) / (self.model["main"][self.n]['W']+self.model["foreign"][self.n]['W'])

    return {
      "confusion_matrix": confusion_matrix,
      "accuracy": accuracy,
      "neutral_rate": neutral_rate,
      "precision": {
        "main": precision_main,
        "foreign": precision_foreign,
        "macro": precision_macro,
        "weighted": precision_weighted
      },
      "recall": {
        "main": recall_main,
        "foreign": recall_foreign,
        "macro": recall_macro,
        "weighted": recall_weighted
      },
      "f1": {
        "main": f1_main,
        "foreign": f1_foreign,
        "macro": f1_macro,
        "weighted": f1_weighted
      }
    }

  def predict(self, text:str) -> List[Tuple] :
    """
      Predict the language of each word in a given `text` input
      Args:
          text: Input text (single word or sentence)
      Returns:
          A list of tuples: [(word1, lang1, lid_index1), (word2, lang2, lid_index2), ...], where:
          - lang1, lang2, ... are 0 (foreign), 1 (main), or -1 (neutral/unknown)
    """
    # Preprocess the text
    text = preprocess_text(text)
    # Split text into words
    words = text.split()

    preds = []
    for word in words :
      lid_index = self.__calculate_lid_index(word)
      if lid_index > 0 :
        lang = 1
      elif lid_index < 0 :
        lang = 0
      else :
        lang = -1
      preds.append((word, lang, lid_index))
    return preds

  def rank(self, text:str) -> dict :
    """
      Rank each word in a given `text` input by its total probability score for each main and foreign languages
      Args:
          text: Input text (single word or sentence)
      Returns:
          A dictionary: {"main": [score1, score2], "foreign": [score1, score2]}, where:
          - score1, score2, ... are the raw scores for each word
    """
    # Preprocess the text
    text = preprocess_text(text)
    # Split text into words
    words = text.split()

    # Compute probability scores for each word
    main_total_p_scores = []
    foreign_total_p_scores = []
    for word in words :
      # Get probabilities for the word in both languages
      main_total_p = self.__calculate_lang_proba(word, "main")
      foreign_total_p = self.__calculate_lang_proba(word, "foreign")
      # Append total probability scores
      main_total_p_scores.append(main_total_p)
      foreign_total_p_scores.append(foreign_total_p)
    return {
      "main": main_total_p_scores,
      "foreign": foreign_total_p_scores
    }

  def __get_frequency_distribution(self, tokens:list) -> dict :
    """
      Returns the words frequency distribution
      Output structure:
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

  def __generate_ngrams(
        self,
        frequency:dict,
        n:int
      ) -> Tuple[Tuple[list,list,list,list,list], Tuple[list,list,list,list,list]] :
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
    single_ngrams_dict = {}
    pre_ngrams_dict = {}
    mid_ngrams_dict = {}
    post_ngrams_dict = {}
    all_ngrams_dict = {}
    for word, freq in frequency.items() :
      # Handle single n-grams
      if len(word) == n :
        all_ngrams_dict[word] = all_ngrams_dict.get(word, 0) + freq
        single_ngrams_dict[word] = single_ngrams_dict.get(word, 0) + freq
      # Handle pre, mid, and post n-grams
      elif len(word) > n :
        ngram_num = len(word) - (n-1)
        for i in range(ngram_num) :
          ngram = word[i:i+n]
          # Update all n-grams
          all_ngrams_dict[ngram] = all_ngrams_dict.get(ngram, 0) + freq
          # Update pre, mid, and post n-grams
          if i==0 :
            pre_ngrams_dict[ngram] = pre_ngrams_dict.get(ngram, 0) + freq
          elif i==ngram_num-1 :
            post_ngrams_dict[ngram] = post_ngrams_dict.get(ngram, 0) + freq
          else :
            mid_ngrams_dict[ngram] = mid_ngrams_dict.get(ngram, 0) + freq

    # Convert dictionaries to lists for output
    single_ngrams, single_ngrams_count = list(single_ngrams_dict.keys()), list(single_ngrams_dict.values())
    pre_ngrams, pre_ngrams_count = list(pre_ngrams_dict.keys()), list(pre_ngrams_dict.values())
    mid_ngrams, mid_ngrams_count = list(mid_ngrams_dict.keys()), list(mid_ngrams_dict.values())
    post_ngrams, post_ngrams_count = list(post_ngrams_dict.keys()), list(post_ngrams_dict.values())
    all_ngrams, all_ngrams_count = list(all_ngrams_dict.keys()), list(all_ngrams_dict.values())
    return (
      (single_ngrams, pre_ngrams, mid_ngrams, post_ngrams, all_ngrams),
      (single_ngrams_count, pre_ngrams_count, mid_ngrams_count, post_ngrams_count, all_ngrams_count)
    )

  def __save_ngrams_to_model(
        self,
        n: int,
        main_ngrams:Tuple[list,list,list,list,list],
        main_ngrams_count:Tuple[list,list,list,list,list],
        foreign_ngrams:Tuple[list,list,list,list,list],
        foreign_ngrams_count:Tuple[list,list,list,list,list],
      ) -> None :
    """
      Store n-grams and the log of their smoothed probas to the model object
      Args:
          main_ngrams: Tuple of lists containing main language n-grams (single, pre, mid, post, all).
          main_ngram_count: Tuple of lists containing the counts for main language n-grams.
          foreign_ngrams: Tuple of lists containing foreign language n-grams (single, pre, mid, post, all).
          foreign_ngram_count: Tuple of lists containing the counts for foreign language n-grams.
    """
    # Calculate lang weights (for evaluation metric)
    N_main = sum(main_ngrams_count[-1])
    N_foreign = sum(foreign_ngrams_count[-1])
    N = N_main + N_foreign
    K = 2
    W_main = N/(K*N_main)
    W_foreign = N/(K*N_foreign)
    self.model["main"][n]['N'] = N_main; self.model["main"][n]['W'] = W_main
    self.model["foreign"][n]['N'] = N_foreign; self.model["foreign"][n]['W'] = W_foreign
    # Theoretically determine vocab size V (number of all possible n-grams) of each lang for smoothing
    main_possible_chars = set()
    foreign_possible_chars = set()
    for ngram in main_ngrams[-1] :
      main_possible_chars.update(set(ngram))
    for ngram in foreign_ngrams[-1] :
      foreign_possible_chars.update(set(ngram))
    V_main = len(main_possible_chars)**n
    V_foreign = len(foreign_possible_chars)**n
    self.model["main"][n]['V'] = V_main
    self.model["foreign"][n]['V'] = V_foreign
    # Store n-grams and their smoothed probas, then save the model
    k = self.model['k']
    self.model["main"][n]["single"] = {ngram: log((ngram_count+k)/(N_main+k*V_main)) for ngram, ngram_count in zip(main_ngrams[0], main_ngrams_count[0])}
    self.model["main"][n]["pre"] = {ngram: log((ngram_count+k)/(N_main+k*V_main)) for ngram, ngram_count in zip(main_ngrams[1], main_ngrams_count[1])}
    self.model["main"][n]["mid"] = {ngram: log((ngram_count+k)/(N_main+k*V_main)) for ngram, ngram_count in zip(main_ngrams[2], main_ngrams_count[2])}
    self.model["main"][n]["post"] = {ngram: log((ngram_count+k)/(N_main+k*V_main)) for ngram, ngram_count in zip(main_ngrams[3], main_ngrams_count[3])}
    self.model["main"][n]["all"] = {ngram: log((ngram_count+k)/(N_main+k*V_main)) for ngram, ngram_count in zip(main_ngrams[4], main_ngrams_count[4])}
    self.model["foreign"][n]["single"] = {ngram: log((ngram_count+k)/(N_foreign+k*V_foreign)) for ngram, ngram_count in zip(foreign_ngrams[0], foreign_ngrams_count[0])}
    self.model["foreign"][n]["pre"] = {ngram: log((ngram_count+k)/(N_foreign+k*V_foreign)) for ngram, ngram_count in zip(foreign_ngrams[1], foreign_ngrams_count[1])}
    self.model["foreign"][n]["mid"] = {ngram: log((ngram_count+k)/(N_foreign+k*V_foreign)) for ngram, ngram_count in zip(foreign_ngrams[2], foreign_ngrams_count[2])}
    self.model["foreign"][n]["post"] = {ngram: log((ngram_count+k)/(N_foreign+k*V_foreign)) for ngram, ngram_count in zip(foreign_ngrams[3], foreign_ngrams_count[3])}
    self.model["foreign"][n]["all"] = {ngram: log((ngram_count+k)/(N_foreign+k*V_foreign)) for ngram, ngram_count in zip(foreign_ngrams[4], foreign_ngrams_count[4])}

  def __get_ngram_proba(self, ngram:str, lang:str, pos:str) -> float :
    """
      Get the *probability* of an n-gram in the specified language and position.
      If the n-gram is not found, recursively back off to lower-order n-grams.
      If the lower-order n-gram (downto unigram) is still not found:
        - k != 0, return smoothed probability; else
        - k == 0, return epsilon
    """
    k = self.model['k']
    # Calculate smoothed probability for unseen n-grams
    p_unseen_ngram = k / (self.model[lang][len(ngram)]['N'] + k * self.model[lang][len(ngram)]['V'])
    # Check if the n-gram exists in the current position
    log_p_ngram = self.model[lang][len(ngram)][pos].get(ngram, None)
    # If the n-gram is found, return its probability
    if log_p_ngram is not None :
      return exp(log_p_ngram)
    # If the n-gram is not found :
    if k!=0 :
      # If k!=0 (smoothed), return p_unseen_ngram
      return p_unseen_ngram
    else :
      # If k==0 (not smoothed), return epsilon
      return self.epsilon

  def __get_ngram_log_proba(self, ngram:str, lang:str, pos:str) -> float :
    """
      Get the *log probability* of an n-gram in the specified language and position.
      If the n-gram is not found, recursively back off to lower-order n-grams.
      If the lower-order n-gram (downto unigram) is still not found:
        - k != 0, return log smoothed probability; else
        - k == 0, return log epsilon
    """
    k = self.model['k']
    # Calculate smoothed probability for unseen n-grams
    p_unseen_ngram = k / (self.model[lang][len(ngram)]['N'] + k * self.model[lang][len(ngram)]['V'])
    # Check if the n-gram exists in the current position
    log_p_ngram = self.model[lang][len(ngram)][pos].get(ngram, None)
    # If the n-gram is found, return its log probability
    if log_p_ngram is not None :
      return log_p_ngram
    # If the n-gram is not found :
    if k!=0 :
      # If k!=0 (smoothed), return log(p_unseen_ngram)
      return log(p_unseen_ngram)
    else :
      # If k==0 (not smoothed), return log(epsilon)
      return log(self.epsilon)

  def __get_ngram_pos(self, i:int, ngram_num:int) -> str :
    """
      Get n-gram position based on its index and number of n-gram to be observed.
    """
    if i==0 :
      if i==ngram_num-1 :
        return "single"
      else :
        return "pre"
    elif i==ngram_num-1 :
      return "post"
    else :
      return "mid"

  def __calculate_lid_index(self, word:str) -> float :
    """
      Calculate the Language Identification (LID) index for a word using normalized log-odds ratios.
      Args:
          word: Input word to calculate the LID index for.
      Returns:
          The LID index, where:
          - Positive values indicate the main language.
          - Negative values indicate the foreign language.
          - Zero indicates a neutral/unknown language.
    """
    word_length = len(word)
    # If the word is shorter than n, use the highest possible n-gram order
    n = min(word_length, self.n)
    # Generate all n-grams for the word
    ngram_num = word_length - (n-1)
    lid_index = 0.
    for i in range(ngram_num) :
      ngram = word[i:i+n]
      pos = self.__get_ngram_pos(i, ngram_num)
      main = self.__get_ngram_log_proba(ngram, "main", pos)
      foreign = self.__get_ngram_log_proba(ngram, "foreign", pos)
      # Get log probas for the n-gram in both languages, add for main and subtract for foreign
      lid_index += main - foreign
    # Normalize the LID index by the number of n-grams
    lid_index /= ngram_num
    return lid_index

  def __calculate_lang_proba(self, word:str, lang:str) -> float :
    """
      Compute the probability of a word in the specified language.
      Args:
        word: Word input.
        lang: Language ("main" or "foreign").
      Returns:
        The probability of the word being in the specified language.
    """
    word_length = len(word)
    # If the word is shorter than n, use the highest possible n-gram order
    n = min(word_length, self.n)
    # Generate all n-grams for the word
    ngram_num = word_length - (n-1)
    log_p_lang = 0. # Use log probabilities to avoid underflow
    for i in range(ngram_num) :
      ngram = word[i:i+n]
      # Get log probas for the n-gram in the specified language
      log_p_ngram = self.__get_ngram_log_proba(ngram, lang, i, ngram_num)
      log_p_lang += log(log_p_ngram)
    # Convert back to the probability value
    return exp(log_p_lang)
