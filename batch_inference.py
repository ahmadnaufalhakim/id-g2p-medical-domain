from argparse import ArgumentParser, Namespace
import csv
import os
import random
from typing import List, Tuple

from modules.prn_gen.src.model import G2P, UNK_TOKEN
from modules.word_type_classifier.src.model import LID
from utils import gen_ngram_candidates, split_g2p_config, verify_args, preprocess_text

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(CURR_DIR, "input")
OUTPUT_DIR = os.path.join(CURR_DIR, "output")

def setup_params(config:Namespace) -> Tuple[str, List[G2P], LID, List[G2P], List[G2P]] :
  mode = config.mode
  en_id_g2ps = None
  lid = None
  en_g2ps = None
  id_g2ps = None
  if mode == "joint" :
    tri_config = split_g2p_config(config=config, prefix="en_id", order="tri")
    bi_config = split_g2p_config(config=config, prefix="en_id", order="bi")
    uni_config = split_g2p_config(config=config, prefix="en_id", order="uni")
    tri_g2p = G2P(config=tri_config)
    bi_g2p = G2P(config=bi_config)
    uni_g2p = G2P(config=uni_config)
    en_id_g2ps = [uni_g2p, bi_g2p, tri_g2p]
  elif mode == "separate" :
    if not (hasattr(config, "alg") and config.alg in ["ngram", "svm", "nb"]) :
      raise ValueError("Invalid word type classifier algorithm. Choose 'ngram', 'svm', or 'nb'.")
    lid = LID(config=config)
    en_tri_config = split_g2p_config(config=config, prefix="en", order="tri")
    en_bi_config = split_g2p_config(config=config, prefix="en", order="bi")
    en_uni_config = split_g2p_config(config=config, prefix="en", order="uni")
    id_tri_config = split_g2p_config(config=config, prefix="id", order="tri")
    id_bi_config = split_g2p_config(config=config, prefix="id", order="bi")
    id_uni_config = split_g2p_config(config=config, prefix="id", order="uni")
    en_tri_g2p = G2P(config=en_tri_config)
    en_bi_g2p = G2P(config=en_bi_config)
    en_uni_g2p = G2P(config=en_uni_config)
    id_tri_g2p = G2P(config=id_tri_config)
    id_bi_g2p = G2P(config=id_bi_config)
    id_uni_g2p = G2P(config=id_uni_config)
    en_g2ps = [en_uni_g2p, en_bi_g2p, en_tri_g2p]
    id_g2ps = [id_uni_g2p, id_bi_g2p, id_tri_g2p]
  return (mode, en_id_g2ps, lid, en_g2ps, id_g2ps)

def batch_inference(
      input_filename:str,
      output_filename:str,
      mode:str,
      en_id_g2ps:List[G2P] = None,
      lid:LID = None,
      en_g2ps:List[G2P] = None,
      id_g2ps:List[G2P] = None,
    ) -> None :
  with open(os.path.join(INPUT_DIR, input_filename)) as f_read, \
       open(os.path.join(OUTPUT_DIR, output_filename), 'w') as f_write :
    # Check header
    sample = f_read.read(1024)
    has_header = csv.Sniffer().has_header(sample)
    f_read.seek(0)

    # Prepare csv reader and writer
    csv_reader = csv.reader(f_read)
    csv_writer = csv.writer(f_write)
    if has_header :
      next(csv_reader)

    # Process all input rows
    for i, row in enumerate(csv_reader) :
      print(f"processing row {i}: {row}")
      inp = preprocess_text(row[0])
      phonemes_list = []

      if mode == "joint" :
        langs = [None]*len(inp.split())
      elif mode == "separate" :
        langs = lid(inp)

      for word, lang in zip(inp.split(), langs) :
        ngram_order = [3, 2, 1]
        ngram_candidates = gen_ngram_candidates(word)
        found_valid_ngram = False # Flag to track if any ngram order indexing is valid
        if mode == "joint" :
          for order in ngram_order :
            indexes = [en_id_g2ps[order-1].GRAPHEME2INDEX.get(grapheme, UNK_TOKEN) for grapheme in ngram_candidates[order-1]]
            if (ngram_candidates[order-1]) and (UNK_TOKEN not in indexes) and (en_id_g2ps[order-1] is not None) :
              phonemes, attentions = en_id_g2ps[order-1](word)
              found_valid_ngram = True
              break
          # If no valid ngram was found
          if not found_valid_ngram :
            for order in ngram_order :
              indexes = [en_id_g2ps[order-1].GRAPHEME2INDEX.get(grapheme, UNK_TOKEN) for grapheme in ngram_candidates[order-1]]
              if (ngram_candidates[order-1]) and (en_id_g2ps[order-1] is not None) :
                phonemes, attentions = en_id_g2ps[order-1](word)
                break
        elif mode == "separate" :
          lang = "en" if lang == 0 or (lang not in [0, 1] and random.random()<.5) else "id"
          g2ps = en_g2ps if lang == "en" else id_g2ps
          for order in ngram_order :
            indexes = [g2ps[order-1].GRAPHEME2INDEX.get(grapheme, UNK_TOKEN) for grapheme in ngram_candidates[order-1]]
            if (ngram_candidates[order-1]) and (UNK_TOKEN not in indexes) and (g2ps[order-1] is not None) :
              phonemes, attentions = g2ps[order-1](word)
              found_valid_ngram = True
              break
          # If no valid ngram was found
          if not found_valid_ngram :
            for order in ngram_order :
              indexes = [g2ps[order-1].GRAPHEME2INDEX.get(grapheme, UNK_TOKEN) for grapheme in ngram_candidates[order-1]]
              if (ngram_candidates[order-1]) and (g2ps[order-1] is not None) :
                phonemes, attentions = g2ps[order-1](word)
                break

        phonemes[0].remove("<EOS>")
        phonemes_list.append(phonemes[0])
      csv_writer.writerow([row[0], ' '.join([' '.join(phonemes) for phonemes in phonemes_list])])

if __name__ == "__main__" :
  # First get all arguments (including --input/--output)
  parser = ArgumentParser()
  parser.add_argument("--input", help="csv input filename", required=False)
  parser.add_argument("--output", help="csv output filename", required=False)

  # Let verify_args handle the rest (it will use parse_known_args internally)
  io_args, remaining_args = parser.parse_known_args()

  config = verify_args(remaining_args=remaining_args)
  mode, en_id_g2ps, lid, en_g2ps, id_g2ps = setup_params(config)

  input_filename = io_args.input if io_args.input is not None else "test.csv"
  if io_args.output is None :
    base = os.path.splitext(input_filename)[0]
    output_filename_suffix = f"_{mode}{f'_{config.alg}' if mode=='separate' else ''}"
    output_filename = f"{base}{output_filename_suffix}.csv"

  batch_inference(
    input_filename,
    output_filename,
    mode,
    en_id_g2ps=en_id_g2ps,
    lid=lid,
    en_g2ps=en_g2ps,
    id_g2ps=id_g2ps,
  )
