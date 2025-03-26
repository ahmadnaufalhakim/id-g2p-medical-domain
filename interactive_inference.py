from argparse import (
  ArgumentParser,
  Namespace
)
import random

from modules.prn_gen.src.model import G2P
from modules.word_type_classifier.src.model import LID

def add_lang_specific_args(parser:ArgumentParser, prefix:str = None) -> None :
  """
    Helper to add language-specific arguments (e.g., --en_grp_type, --id_grp_type).
    Default prefix is set to '' for joint mode.
  """
  parser.add_argument(
    f"--{f'{prefix}_' if prefix else ''}mdl_prefix",
    help=f"{f'{prefix} ' if prefix else ''} model filename prefix",
    nargs='?', const=1, default=""
  )
  parser.add_argument(
    f"--{f'{prefix}_' if prefix else ''}grp_type",
    help=f"{f'{prefix} ' if prefix else ''} model grapheme type",
    choices=["unigram", "bigram", "trigram"],
    required=True
  )
  parser.add_argument(
    f"--{f'{prefix}_' if prefix else ''}weight_decay",
    help=f"{f'{prefix} ' if prefix else ''} model weight decay",
    choices=["1e_5", "1e_4"],
    nargs='?', const=1, default="1e_5"
  )
  parser.add_argument(
    f"--{f'{prefix}_' if prefix else ''}attn_model",
    help=f"{f'{prefix} ' if prefix else ''} model attention type",
    choices=["dot", "general", "concat"],
    default="dot"
  )
  parser.add_argument(f"--{f'{prefix}_' if prefix else ''}emb_dim", help=f"{f'{prefix} ' if prefix else ''} model embedding dimension", type=int, required=True)
  parser.add_argument(f"--{f'{prefix}_' if prefix else ''}hidden_size", help=f"{f'{prefix} ' if prefix else ''} model hidden layer size", type=int, required=True)
  parser.add_argument(f"--{f'{prefix}_' if prefix else ''}n_layers", help=f"{f'{prefix} ' if prefix else ''} model number of hidden layers", type=int, default=1)

def split_config(config:Namespace, prefix:str) -> None :
  """
    Extract prefixed arguments into a new Namespace.
  """
  new_config = Namespace()
  for key, value in vars(config).items() :
    if key.startswith(f"{prefix}_") :
      clean_key = key[len(prefix)+1:]
      setattr(new_config, clean_key, value)
    elif not hasattr(new_config, key) :
      setattr(new_config, key, value)
  return new_config

def interactive_inference(
      config:Namespace
    ) :
  if not (hasattr(config, "mode") and config.mode in ["joint", "separate"]) :
    raise ValueError("Invalid mode. Choose 'joint' or 'separate'.")
  mode = config.mode
  if mode == "joint" :
    if not (hasattr(config, "lang") and config.lang == "en_id") :
      raise ValueError("Invalid lang for joint mode. Use 'en_id'.")
    g2p = G2P(config=config)
    while True :
      inp = input("\nInput (enter 'exit' to quit): ").strip()
      if inp.lower() == "exit" :
        confirm = input("Are you sure? (y/n): ").strip().lower()
        if confirm == 'y' :
          break
      phonemes, attentions = g2p(inp)
      print(phonemes)
  elif mode == "separate" :
    if not (hasattr(config, "alg") and config.alg in ["ngram", "svm"]) :
      raise ValueError("Invalid word type classifier algorithm. Choose 'ngram' or 'svm'.")
    lid = LID(config=config)

    en_config = split_config(config, "en")
    setattr(en_config, "lang", "en")
    en_g2p = G2P(config=en_config)

    id_config = split_config(config, "id")
    setattr(id_config, "lang", "id")
    id_g2p = G2P(config=id_config)

    while True :
      inp = input("\nInput (enter 'exit' to quit): ").strip()
      if inp.lower() == "exit" :
        confirm = input("Are you sure? (y/n): ").strip().lower()
        if confirm == 'y' :
          break
      decoded_phonemes = []
      langs = lid(inp)
      for word, lang in zip(inp.split(), langs) :
        if lang == 0 :
          phonemes, attentions = en_g2p(word)
        elif lang == 1 :
          phonemes, attentions = id_g2p(word)
        elif random.random() < .5 :
          phonemes, attentions = en_g2p(word)
        else :
          phonemes, attentions = id_g2p(word)
        decoded_phonemes.append(phonemes)
      print(langs)
      print(decoded_phonemes)

if __name__ == "__main__" :
  parser = ArgumentParser()
  subparsers = parser.add_subparsers(dest="mode", required=True)

  # Joint mode (just G2P)
  joint_parser = subparsers.add_parser("joint", help="Unified en_id G2P model")
  add_lang_specific_args(parser=joint_parser)
  # Separate mode (LID + dual G2P)
  separate_parser = subparsers.add_parser("separate", help="LID + language-specific G2P")
  separate_parser.add_argument(
    "--alg",
    help="LID algorithm",
    choices=["ngram", "svm"],
    required=True
  )
  # Conditional args for --alg
  ngram_group = separate_parser.add_argument_group("ngram_args", "n-gram LID options")
  ngram_group.add_argument("--n", help="n-gram order", type=int)
  ngram_group.add_argument("--k", help="Smoothing parameter")
  svm_group = separate_parser.add_argument_group("svm_args", "SVM LID options")
  svm_group.add_argument("--kernel", help="SVM kernel", choices=["linear", "rbf", "sigmoid"])
  # G2P args
  add_lang_specific_args(separate_parser, "en")
  add_lang_specific_args(separate_parser, "id")
  args = parser.parse_args()
  print(args)

  interactive_inference(config=args)
