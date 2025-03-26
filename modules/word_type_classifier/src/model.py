from argparse import Namespace
import joblib
import numpy as np
import os
from sklearn.svm import LinearSVC, SVC
from typing import List, Union

from .ngram_model import NGramClassifier

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURR_DIR, "..", "models")

class LID :
  def __init__(self, config:Namespace) -> None :
    self.alg = config.alg
    self.clf:Union[NGramClassifier, LinearSVC, SVC] = None
    if self.alg == "ngram" :
      assert hasattr(config, 'n'), "Invalid n n-gram value"
      n = int(config.n)
      k = float(config.k) if hasattr(config, 'k') and config.k is not None else None
      self.clf = NGramClassifier(n=n, k=k)
      self.clf.load()
    elif self.alg == "svm" :
      assert hasattr(config, "kernel") and config.kernel in ["linear", "rbf", "sigmoid"], "Invalid kernel type. Choose 'linear', 'rbf', or 'sigmoid'."
      kernel = config.kernel
      self.clf = joblib.load(os.path.join(MODELS_DIR, f"svms/pipeline-{kernel}.pkl"))

  def __call__(self, input:str) -> Union[List[int], np.ndarray] :
    if self.alg == "ngram" :
      lang_preds = [pred[1] for pred in self.clf.predict(input)]
    elif self.alg == "svm" :
      words = input.split()
      lang_preds = self.clf.predict(words)
    return lang_preds
