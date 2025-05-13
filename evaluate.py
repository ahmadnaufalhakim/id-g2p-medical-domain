from argparse import Namespace
from collections import Counter, defaultdict
from jiwer import wer
import os
import random
from typing import List, Tuple

from modules.prn_gen.src.model import G2P, UNK_TOKEN
from modules.word_type_classifier.src.model import LID
from utils import gen_ngram_candidates, split_g2p_config, verify_args, preprocess_text

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "modules", "prn_gen", "data")

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

def evaluate(
      val_pairs:List[List[str]],
      mode:str,
      en_id_g2ps:List[G2P] = None,
      lid:LID = None,
      en_g2ps:List[G2P] = None,
      id_g2ps:List[G2P] = None
    ) -> float :
  total_per = 0.
  # For handling PER calculation for words with multiple pronunciations
  words = [pair[0] for pair in val_pairs]
  word_counts = Counter(words)
  dupe_words = [item for item, count in word_counts.items() if count>1]
  multiple_pronunciation_dict = defaultdict(list)
  for word, pron, lang in val_pairs :
    if word in dupe_words :
      multiple_pronunciation_dict[word].append(pron)
  multiple_pronunciation_dict = dict(multiple_pronunciation_dict)

  # Iterate and process all words in the val pairs
  for pair in val_pairs :
    word, arpabet_phoneme_sequence, ref_lang = pair
    word = preprocess_text(word)

    ngram_order = [3, 2, 1]
    ngram_candidates = gen_ngram_candidates(word)
    found_valid_ngram = False # Flag to track if any ngram order indexing is valid
    if mode == "joint" :
      for order in ngram_order :
        indexes = [en_id_g2ps[order-1].GRAPHEME2INDEX.get(grapheme, UNK_TOKEN) for grapheme in ngram_candidates[order-1]]
        if (ngram_candidates[order-1]) and (UNK_TOKEN not in indexes) and (en_id_g2ps[order-1] is not None) :
          output_phonemes, _ = en_id_g2ps[order-1](word)
          found_valid_ngram = True
          break
      # If no valid ngram was found
      if not found_valid_ngram :
        for order in ngram_order :
          indexes = [en_id_g2ps[order-1].GRAPHEME2INDEX.get(grapheme, UNK_TOKEN) for grapheme in ngram_candidates[order-1]]
          if (ngram_candidates[order-1]) and (en_id_g2ps[order-1] is not None) :
            output_phonemes, _ = en_id_g2ps[order-1](word)
            break
    elif mode == "separate" :
      hyp_langs = lid(word)
      hyp_lang = "en" if hyp_langs[0] == 0 or (hyp_langs[0] not in [0, 1] and random.random()<.5) else "id"
      g2ps = en_g2ps if hyp_lang == "en" else id_g2ps
      for order in ngram_order :
        indexes = [g2ps[order-1].GRAPHEME2INDEX.get(grapheme, UNK_TOKEN) for grapheme in ngram_candidates[order-1]]
        if (ngram_candidates[order-1]) and (UNK_TOKEN not in indexes) and (g2ps[order-1] is not None) :
          output_phonemes, _ = g2ps[order-1](word)
          found_valid_ngram = True
          break
      # If no valid ngram was found
      if not found_valid_ngram :
        for order in ngram_order :
          indexes = [g2ps[order-1].GRAPHEME2INDEX.get(grapheme, UNK_TOKEN) for grapheme in ngram_candidates[order-1]]
          if (ngram_candidates[order-1]) and (g2ps[order-1] is not None) :
            output_phonemes, _ = g2ps[order-1](word)
            break

    # Calculate WER
    output_phonemes[0].remove("<EOS>")
    if word in dupe_words :
      curr_wer = min([
        wer(
          arpabet_phoneme_seq,
          ' '.join(output_phonemes[0])
        ) for arpabet_phoneme_seq in multiple_pronunciation_dict[word]
      ])
    else :
      curr_wer = wer(
        arpabet_phoneme_sequence,
        ' '.join(output_phonemes[0])
      )
    total_per += curr_wer
    if mode == "joint" :
      print(word, f"ref:{arpabet_phoneme_sequence.split()}", f"hyp:{output_phonemes[0]}", curr_wer, f"order_{order}")
    elif mode == "separate" :
      print(word, f"ref:{arpabet_phoneme_sequence.split()}", f"hyp:{output_phonemes[0]}", curr_wer, f"ref:{ref_lang}", f"hyp:{hyp_lang}", f"order_{order}")
  return total_per*100/len(val_pairs)

if __name__ == "__main__" :
  with open(os.path.join(DATA_DIR, "en_ma/test_converted.csv")) as f_csv :
    next(f_csv, None)
    val_pairs = [[s.strip('\n') for s in row.split(',')] for row in f_csv]

  config = verify_args()
  mode, en_id_g2ps, lid, en_g2ps, id_g2ps = setup_params(config)
  print(evaluate(
    val_pairs,
    mode,
    en_id_g2ps=en_id_g2ps,
    lid=lid,
    en_g2ps=en_g2ps,
    id_g2ps=id_g2ps,
  ))
