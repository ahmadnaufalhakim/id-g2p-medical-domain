from decimal import Decimal
from math import floor
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from sklearn.model_selection import GridSearchCV
import string
import time

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "..", "data")
MODELS_DIR = os.path.join(CURR_DIR, "..", "models")
if not os.path.exists(MODELS_DIR) :
  os.mkdir(MODELS_DIR)
OUTPUT_DIR = os.path.join(CURR_DIR, "..", "output")
if not os.path.exists(OUTPUT_DIR) :
  os.mkdir(OUTPUT_DIR)

def preprocess_text(text:str) -> str :
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

def extract_tokens_from_file(filepath:str, lowercase:bool=True) -> list :
  fstream = open(filepath, 'r', encoding="UTF_8").read()
  if lowercase :
    fstream = fstream.lower()
  clean_text = preprocess_text(fstream)
  return clean_text.split()

# Helper function to plot n-gram score for hyperparameter searching
def plot_ngram_score(title:str, n:int, **kwargs) -> None :
  global k_list
  plt.figure()
  fig, ax = plt.subplots()
  # # Set x-axis ticks and labels
  plt.title(label=f"{n}-gram {title.title()}")
  plt.ylabel(ylabel=title.title())
  plt.xlabel(xlabel="k smoothing value")
  legends = []
  values = [val for values_list in kwargs.values() for val in values_list]
  for k, v in kwargs.items() :
    plt.plot(k_list, v)
    if k == "macro" or k == "accuracy" :
      # Find the maximum value and its index
      max_value = max(v)
      max_index = v.index(max_value)
      # Plot a red dot at the maximum value
      plt.plot(k_list[max_index], max_value, "ro")
      # Add text box in the middle of the plot showing the minimum value
      y_range = max(values)-min(values)
      offset = .075 * y_range
      plt.text(k_list[max_index]-0.05, max_value-offset, f"{max_value:.4f}", bbox=dict(facecolor="white", alpha=.5))
    if len(kwargs) > 1 :
      legends.append(k)
  if legends : plt.legend(legends)
  plt.savefig(os.path.join(OUTPUT_DIR, f"{n}gram-{'-'.join(title.split())}.png"))
  plt.close()

# Helper functions to log training progress
def as_minutes(seconds:float) -> str :
  minutes = floor(seconds/60)
  seconds -= minutes*60
  return f"{minutes}m {round(seconds, 2)}s"
def time_since(since:float, percent:float) :
  now = time.time()
  seconds = now - since
  eta_seconds = seconds/(percent)
  remaining_seconds = eta_seconds - seconds
  return f"{as_minutes(seconds)} (- {as_minutes(remaining_seconds)})"

# Helper functions to plot SVM score for hyperparameter searching
def get_font_color(bg_color:np.ndarray) :
  """
    Determine font color (white or black) based on the luminance of the background color.
  """
  luminance = np.mean(bg_color[:3])  # Average of RGB values
  return 'white' if luminance < 0.5 else 'black'
def plot_heatmap(
      grid:np.ndarray,
      metric:str,
      C_values:list,
      gamma_values:list,
      coef0:float=None
    ) -> None :
  """
    Plot and save a 2D heatmap for a SVM with RBF or Sigmoid kernel.

    Args:
        grid: 2D array of mean test scores for C and gamma.
        metric: The type of score.
        C_values: List of C values.
        gamma_values: List of gamma values.
        coef0: The fixed coef0 value for this heatmap.
  """
  kernel = "sigmoid" if coef0 is not None else "rbf"
  plt.figure()
  heatmap = plt.imshow(grid, cmap="viridis", aspect="auto")
  cbar = plt.colorbar(heatmap)
  cbar.set_label(f"{metric.title()} Score")
  plt.title(label=f"{metric.title()} Score Heatmap for {kernel.upper() if kernel=='rbf' else kernel.title()} Kernel{f' (coef0={coef0})' if coef0 is not None else ''}")
  plt.xlabel("gamma")
  plt.ylabel('C')
  plt.xticks(np.arange(len(gamma_values)), gamma_values)
  plt.yticks(np.arange(len(C_values)), C_values)
  # Annotate each cell with the score value
  for i in range(len(C_values)) :
    for j in range(len(gamma_values)) :
      bg_color = heatmap.cmap(heatmap.norm(grid[i, j]))
      font_color = get_font_color(bg_color=bg_color)
      plt.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", color=font_color)
  output_file = os.path.join(OUTPUT_DIR, "svms", f"svm-{kernel}{f'-coef0_{coef0}' if coef0 is not None else ''}-{metric}-heatmap.png")
  plt.savefig(output_file, dpi=300, bbox_inches="tight")
  plt.close()
def plot_svm_score(grid_search:GridSearchCV, kernel:str, metric:str) -> None :
  """
    Plot and save a heatmap of mean test scores for SVM hyperparameters.

    Args:
        grid_search: A fitted GridSearchCV object.
        kernel: SVM kernel type ('linear', 'rbf', or 'sigmoid').
        metric: The metric used to score the SVM ('accuracy' or 'f1').
  """
  assert kernel in ["linear", "rbf", "sigmoid"], "Invalid kernel type. Choose 'linear', 'rbf', or 'sigmoid'."
  assert metric in ["accuracy", "f1"], "Invalid metric. Choose 'accuracy' or 'f1'"
  # Extract results
  results = grid_search.cv_results_
  mean_test_scores = results["mean_test_score"]
  params = results["params"]
  # Sort C and gamma values
  C_values = sorted(list(set(param['C'] for param in params)))
  if kernel!="linear" :
    gamma_values = sorted(list(set(param["gamma"] for param in params)))
    if kernel=="sigmoid" :
      coef0_values = sorted(list(set(param["coef0"] for param in params)))

  # Plot based on kernel type
  if kernel=="linear" :
    plt.figure()
    plt.plot(C_values, mean_test_scores, marker='o', linestyle='-', color='b')
    plt.xscale("log")
    plt.xlabel("C (Regularization Parameter)")
    plt.ylabel(f"Mean {metric.title()} Score")
    plt.title(f"{metric.title()} Score for {kernel.title()} Kernel")
    plt.grid(True)
    output_file = os.path.join(OUTPUT_DIR, "svms", f"svm-{kernel}-{metric}-plot.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
  elif kernel=="rbf" :
    scores_grid = np.zeros((len(C_values), len(gamma_values)))
    for i, C in enumerate(C_values) :
      for j, gamma in enumerate(gamma_values) :
        for param, score in zip(params, mean_test_scores) :
          if param['C']==C and param["gamma"]==gamma :
            scores_grid[i, j] = score
            break
    plot_heatmap(
      grid=scores_grid,
      metric=metric,
      C_values=C_values,
      gamma_values=gamma_values,
    )
  else :
    for coef0 in coef0_values :
      scores_grid = np.zeros((len(C_values), len(gamma_values)))
      for i, C in enumerate(C_values) :
        for j, gamma in enumerate(gamma_values) :
          for param, score in zip(params, mean_test_scores) :
            if param['C']==C and param["gamma"]==gamma and param["coef0"]==coef0 :
              scores_grid[i, j] = score
              break
      plot_heatmap(
        grid=scores_grid,
        metric=metric,
        C_values=C_values,
        gamma_values=gamma_values,
        coef0=coef0
      )

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
