import argparse
import os
import torch

from modules.prn_gen.src.model import (
  DEVICE,
  Encoder, Decoder
)

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PRN_GEN_DIR = os.path.join(CURR_DIR, "modules", "prn-gen")
WORD_TYPE_CLASSIFIER_DIR = os.path.join(CURR_DIR, "modules", "word-type-classifier")

