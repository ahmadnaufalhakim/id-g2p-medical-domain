from argparse import (
  ArgumentParser,
  Namespace
)
from itertools import chain
import re
import string
import sys
from typing import List

def add_lang_order_specific_args(parser:ArgumentParser, prefix:str = None) -> None :
  """
    Helper to add language and n-gram order specific arguments (e.g., --en_tri_grp_type, --id_tri_grp_type).
    Default prefix is set to '' for joint mode.
  """
  for order in ["uni", "bi", "tri"] :
    parser.add_argument(
      f"--{f'{prefix}_{order}_' if prefix else f'en_id_{order}_'}mdl_prefix",
      help=f"{f'{prefix}_{order} ' if prefix else f'en_id_{order}'} model filename prefix",
      nargs='?', const=1
    )
    parser.add_argument(
      f"--{f'{prefix}_{order}_' if prefix else f'en_id_{order}_'}grp_type",
      help=f"{f'{prefix}_{order} ' if prefix else f'en_id_{order}'} model grapheme type",
      choices=["unigram", "bigram", "trigram"]
    )
    parser.add_argument(
      f"--{f'{prefix}_{order}_' if prefix else f'en_id_{order}_'}weight_decay",
      help=f"{f'{prefix}_{order} ' if prefix else f'en_id_{order}'} model weight decay",
      choices=["1e_5", "1e_4"],
      nargs='?', const=1, default="1e_5"
    )
    parser.add_argument(
      f"--{f'{prefix}_{order}_' if prefix else f'en_id_{order}_'}attn_model",
      help=f"{f'{prefix}_{order} ' if prefix else f'en_id_{order}'} model attention type",
      choices=["dot", "general", "concat"],
      default="dot"
    )
    parser.add_argument(f"--{f'{prefix}_{order}_' if prefix else f'en_id_{order}_'}emb_dim", help=f"{f'{prefix}_{order} ' if prefix else f'en_id_{order}'} model embedding dimension", type=int)
    parser.add_argument(f"--{f'{prefix}_{order}_' if prefix else f'en_id_{order}_'}hidden_size", help=f"{f'{prefix}_{order} ' if prefix else f'en_id_{order}'} model hidden layer size", type=int)
    parser.add_argument(f"--{f'{prefix}_{order}_' if prefix else f'en_id_{order}_'}n_layers", help=f"{f'{prefix}_{order} ' if prefix else f'en_id_{order}'} model number of hidden layers", type=int, default=1)

def get_default_configs(mode:str) -> Namespace :
  """
    Returns mode-specific default configuration for each language and grapheme type.
  """
  # # LID default hyperparameters config
  # lid_ngram_cfg = {
  #   "alg": "ngram",
  #   'n': 3,
  #   'k': 0.
  # }
  # lid_svm_cfg = {
  #   "alg": "svm",
  #   "kernel": "rbf"
  # }
  # lid_nb_cfg = {
  #   "alg": "nb",
  #   "nb_type": "multinomial"
  # }
  # Embedding dimension-hidden size pairwise hyperparameters for each n-gram order [uni,bi,tri]
  EN_ID_N_EMB_HDN = [("uni",64,128), ("bi",256,128), ("tri",512,64)]
  EN_N_EMB_HDN = [("uni",32,100), ("bi",128,50), ("tri",64,50)]
  ID_N_EMB_HDN = [("uni",64,50), ("bi",64,32), ("tri",128,128)]

  en_id_configs = [{
    f"en_id_{order}_mdl_prefix": "train-",
    f"en_id_{order}_grp_type": f"{order}gram",
    f"en_id_{order}_weight_decay": "1e_5",
    f"en_id_{order}_attn_model": "dot",
    f"en_id_{order}_emb_dim": emb_dim,
    f"en_id_{order}_hidden_size": hidden_size,
    f"en_id_{order}_n_layers": 1,
  } for order, emb_dim, hidden_size in EN_ID_N_EMB_HDN]
  en_configs = [{
    f"en_{order}_mdl_prefix": "train-",
    f"en_{order}_grp_type": f"{order}gram",
    f"en_{order}_weight_decay": "1e_5",
    f"en_{order}_attn_model": "dot",
    f"en_{order}_emb_dim": emb_dim,
    f"en_{order}_hidden_size": hidden_size,
    f"en_{order}_n_layers": 1,
  } for order, emb_dim, hidden_size in EN_N_EMB_HDN]
  id_configs = [{
    f"id_{order}_mdl_prefix": "train-",
    f"id_{order}_grp_type": f"{order}gram",
    f"id_{order}_weight_decay": "1e_5",
    f"id_{order}_attn_model": "dot",
    f"id_{order}_emb_dim": emb_dim,
    f"id_{order}_hidden_size": hidden_size,
    f"id_{order}_n_layers": 1,
  } for order, emb_dim, hidden_size in ID_N_EMB_HDN]

  # lid_config = {} if mode=="joint" else lid_ngram_cfg

  defaults = {
    "mode": mode,
    **dict(chain.from_iterable(d.items() for d in en_id_configs)),
    # **lid_config,
    **dict(chain.from_iterable(d.items() for d in en_configs)),
    **dict(chain.from_iterable(d.items() for d in id_configs)),
  }
  return Namespace(**defaults)

def split_g2p_config(config:Namespace, prefix:str, order:str) -> Namespace :
  """
    Extract arguments matching both the prefix AND n-gram order.

    Args:
        config: Input Namespace (e.g., from argparse)
        prefix: Language prefix ('en', 'id', 'en_id')
        ngram_order: N-gram order ('uni', 'bi', 'tri')

    Returns:
        Namespace: Filtered config with cleaned keys (prefix/order removed).
                   Non-prefixed args (e.g., 'mode') are included unchanged.
    """
  new_config = Namespace()
  target_prefix = f"{prefix}_{order}_" # e.g., "en_tri_"
  for key, value in vars(config).items() :
    # Matches {prefix}_{order}_* (e.g., "en_tri_mdl_prefix")
    if key.startswith(target_prefix) :
      clean_key = key[len(target_prefix):] # Remove "en_tri_"
      setattr(new_config, clean_key, value)
  setattr(new_config, "lang", prefix)
  return new_config

def verify_args(remaining_args:list = None) -> Namespace :
  if remaining_args is None :
    remaining_args = sys.argv[1:]
  # --- First Pass: Only check --mode and --use-defaults ---
  stage1_parser = ArgumentParser(add_help=False)
  stage1_parser.add_argument("--mode", choices=["joint", "separate"], required=True)
  stage1_parser.add_argument("--use-defaults", action="store_true")
  stage1_args, remaining_args = stage1_parser.parse_known_args(remaining_args)

  # --- Second Pass: Full parsing (if no defaults) ---
  if stage1_args.use_defaults :
    args = get_default_configs(stage1_args.mode)

    # Create parser for potential override
    override_parser = ArgumentParser(add_help=False)

    if stage1_args.mode == "joint" :
      add_lang_order_specific_args(parser=override_parser)
    elif stage1_args.mode == "separate" :
      # Check if --alg is in remaining_args
      has_alg = any(arg.startswith("--alg") for arg in remaining_args)
      if not has_alg :
        override_parser.error("the following argument is required when --mode=separate and --use-defaults: --alg {ngram,svm,nb}")

      # LID args
      override_parser.add_argument("--alg", choices=["ngram", "svm", "nb"], required=True)
      # G2P args
      add_lang_order_specific_args(override_parser, "en")
      add_lang_order_specific_args(override_parser, "id")

    # Parse only the overriding args
    override_args = override_parser.parse_args(remaining_args)
    # Apply overrides to defaults
    for key, value in vars(override_args).items() :
      # Apply default LID algorithms
      if key == "alg" :
        if value == "ngram" :
          setattr(args, 'n', 3)
          setattr(args, 'k', .0)
        elif value == "svm" :
          setattr(args, "kernel", "rbf")
        elif value == "nb" :
          setattr(args, "nb_type", "multinomial")
      if value is not None :
        setattr(args, key, value)
  else :
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Joint mode (just G2P)
    joint_parser = subparsers.add_parser("joint", help="Unified en_id G2P model")
    add_lang_order_specific_args(parser=joint_parser)
    # Separate mode (LID + dual G2P)
    separate_parser = subparsers.add_parser("separate", help="LID + language-specific G2P")
    separate_parser.add_argument(
      "--alg",
      help="LID algorithm",
      choices=["ngram", "svm", "nb"],
      required=True
    )
    # Conditional args for --alg
    ngram_group = separate_parser.add_argument_group("ngram_args", "n-gram LID options")
    ngram_group.add_argument("--n", help="n-gram order", type=int)
    ngram_group.add_argument("--k", help="Smoothing parameter")
    svm_group = separate_parser.add_argument_group("svm_args", "SVM LID options")
    svm_group.add_argument("--kernel", help="SVM kernel", choices=["linear", "rbf", "sigmoid"])
    nb_group = separate_parser.add_argument_group("nb_args", "NB LID options")
    nb_group.add_argument("--nb_type", help="NB type", choices=["bernoulli", "multinomial"])
    # G2P args
    add_lang_order_specific_args(separate_parser, "en")
    add_lang_order_specific_args(separate_parser, "id")
    args = parser.parse_args()
  return args

def gen_ngram_candidates(word:str) -> List[List[str]] :
  """
    Helper function to generate n-gram candidates for each n-gram order
    e.g., "helper" -> [['h','e','l','p','e','r'],["he","el","lp","pe","er"],["hel","elp","lpe","per"]]
  """
  candidates = []
  for n in range(1, 3+1) :
    ngrams = [word[i:i+n] for i in range(len(word)-(n-1))]
    candidates.append(ngrams)
  return candidates

def preprocess_text(text:str) -> str :
  """
    Returns cleaned text (no dashes, digits, tabs, punctuation, non-alphabetic characters)
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