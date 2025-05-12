from argparse import (
  ArgumentParser,
  Namespace
)
import os
import torch
from typing import List, Tuple

from model import (
  DEVICE,
  Encoder, Decoder
)

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(CURR_DIR, "..", "exp")
MODELS_DIR = ''

SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2
UNK_TOKEN = 3

GRP_TYPE = ''
MAX_LENGTH = 32
INDEX2GRAPHEME = None
GRAPHEME2INDEX = None
INDEX2PHONEME = None
encoder = None
decoder = None

def load_mappings(config:Namespace) -> None :
  print("Loading mappings..")
  global MODELS_DIR
  global GRP_TYPE, INDEX2GRAPHEME, GRAPHEME2INDEX, INDEX2PHONEME

  MODELS_DIR = os.path.join(EXPERIMENTS_DIR, f"{config.lang}/models/{config.grp_type}")
  if not os.path.exists(MODELS_DIR) :
    raise Exception("models directory not found")
  GRP_TYPE = config.grp_type
  print("\tloading id2grp")
  INDEX2GRAPHEME = torch.load(os.path.join(MODELS_DIR, "id2grp.pth"), weights_only=True)
  INDEX2GRAPHEME = {
    k: v.lower() if k not in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN] else v for k, v in INDEX2GRAPHEME.items()
  }
  GRAPHEME2INDEX = {
    v: k for k, v in INDEX2GRAPHEME.items() if k not in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN]
  }
  print("\tloading id2phn")
  INDEX2PHONEME = torch.load(os.path.join(MODELS_DIR, "id2phn.pth"), weights_only=True)

def load_models(config:Namespace) -> None :
  print("Loading models..")
  global encoder, decoder

  mdl_suffix = f"-{f'wdecay_{config.weight_decay}-' if hasattr(config, 'weight_decay') and config.weight_decay else ''}attn_{config.attn_model}-emb_{config.emb_dim}-hddn_{config.hidden_size}-layers_{config.n_layers}-epoch_"
  encoders = sorted([
    f for f in os.listdir(MODELS_DIR) if f.startswith(f"{config.mdl_prefix}encoder{mdl_suffix}")],
    key=lambda fn: int(fn.split("epoch_")[1].split(".pth")[0])
  )
  decoders = sorted([
    f for f in os.listdir(MODELS_DIR) if f.startswith(f"{config.mdl_prefix}decoder{mdl_suffix}")],
    key=lambda fn: int(fn.split("epoch_")[1].split(".pth")[0])
  )
  print(f"{config.mdl_prefix}encoder{mdl_suffix}")
  print(f"{config.mdl_prefix}decoder{mdl_suffix}")
  assert len(encoders)>0 and len(decoders)>0, \
    f"""No encoder and decoder found with the following parameters:
    - language: {config.lang}
    {f"- model prefix: {config.mdl_prefix}" if hasattr(config, "mdl_prefix") and config.mdl_prefix else ''}
    - grapheme type: {config.grp_type}
    {f"- weight decay: {config.weight_decay}" if hasattr(config, "weight_decay") and config.weight_decay else ''}
    - attention scoring method: {config.attn_model}
    - embedding dimension: {config.emb_dim}
    - hidden layer size: {config.hidden_size}
    - number of layers: {config.n_layers}"""

  # Get best encoder and decoder path
  best_encoder = encoders[-1]
  best_decoder = decoders[-1]
  print(f"\tfound best encoder: {best_encoder}")
  print(f"\tfound best decoder: {best_decoder}")
  # Initialize models
  print("\tinitializing models")
  encoder = Encoder(len(INDEX2GRAPHEME), config.emb_dim, config.hidden_size, config.n_layers)
  decoder = Decoder(config.attn_model, config.emb_dim, config.hidden_size, len(INDEX2PHONEME), config.n_layers)
  # Load weights
  print("\tloading encoder's state dict")
  encoder.load_state_dict(torch.load(os.path.join(MODELS_DIR, best_encoder), map_location=DEVICE, weights_only=True))
  print("\tencoder's state dict loaded")
  print("\tloading decoder's state dict")
  decoder.load_state_dict(torch.load(os.path.join(MODELS_DIR, best_decoder), map_location=DEVICE, weights_only=True))
  print("\tdecoder's state dict loaded")
  # Move to GPU if available
  encoder = encoder.to(DEVICE).eval()
  decoder = decoder.to(DEVICE).eval()

def word_to_tensor(word:str) -> torch.Tensor :
  assert GRP_TYPE in ["unigram", "bigram", "trigram"]
  if GRP_TYPE == "unigram" :
    graphemes_list = [[*word]]
  elif GRP_TYPE == "bigram" :
    if len(word) < 2 :
      graphemes_list = [[word]]
    else :
      graphemes_list = [[word[i:i+2] for i in range(len(word)-1)], [*word]]
  elif GRP_TYPE == "trigram" :
    if len(word) < 3 :
      if len(word) < 2 :
        graphemes_list = [[word]]
      else :
        graphemes_list = [[word[i:i+2] for i in range(len(word)-1)], [*word]]
    else :
      graphemes_list = [[word[i:i+3] for i in range(len(word)-2)], [word[i:i+2] for i in range(len(word)-1)], [*word]]
  # Thoroughly check if there are no UNK_TOKEN in the indexes of all graphemes list
  for graphemes in graphemes_list :
    indexes = [GRAPHEME2INDEX.get(grapheme, UNK_TOKEN) for grapheme in graphemes] + [EOS_TOKEN]
    # If there are no UNK_TOKEN in the indexes, immediately return the tensor of the indexes
    if UNK_TOKEN not in indexes :
      tensor = torch.LongTensor(indexes).view(-1, 1).to(DEVICE)
      return tensor

  # If all possible indexes contain UNK_TOKEN, defaults to the indexes of the model's grapheme type
  if GRP_TYPE == "unigram" :
    graphemes = [*word]
  elif GRP_TYPE == "bigram" :
    graphemes = [word[i:i+2] for i in range(len(word)-1)] if len(word)>=2 else [word]
  elif GRP_TYPE == "trigram" :
    graphemes = [word[i:i+3] for i in range(len(word)-2)] if len(word)>=3 else [word]
  indexes = [GRAPHEME2INDEX.get(grapheme, UNK_TOKEN) for grapheme in graphemes] + [EOS_TOKEN]
  tensor = torch.LongTensor(indexes).view(-1, 1).to(DEVICE)
  return tensor

def infer(
      word:str,
      max_length:int = None,
      with_attention:bool = False
    ) -> Tuple[List[str], torch.Tensor] :
  input_tensor = word_to_tensor(word)
  if max_length is None :
    max_length = MAX_LENGTH
  # Run through encoder
  encoder_hidden = encoder.init_hidden()
  encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
  # Prepare decoder inputs
  decoder_input = torch.LongTensor([[SOS_TOKEN]]).to(DEVICE)
  decoder_context = torch.zeros(1, decoder.hidden_size).to(DEVICE)
  decoder_hidden = encoder_hidden
  decoder_attentions = torch.zeros(max_length, max_length)
  decoded_phonemes = []
  for di in range(max_length) :
    decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(
      decoder_input, decoder_context, decoder_hidden, encoder_outputs
    )
    if with_attention :
      decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data
    # Choose top output
    topv, topi = decoder_output.data.topk(1)
    ni = topi[0][0]
    if ni.item() in [EOS_TOKEN, PAD_TOKEN] :
      decoded_phonemes.append("<EOS>")
      break
    else :
      decoded_phonemes.append(INDEX2PHONEME[ni.item()])
    # Next decoder input is last token
    decoder_input = torch.LongTensor([[ni.item()]]).to(DEVICE)
  return decoded_phonemes, decoder_attentions[:di+1, 1:len(encoder_outputs)]

def infer_sentence(sentence:str) -> Tuple[List[List[str]], List[torch.Tensor]] :
  words = sentence.split()
  decoded_phonemes_list = []
  decoder_attentions_list = []
  for word in words :
    decoded_phonemes, decoder_attns = infer(word.lower())
    decoded_phonemes = post_process(decoded_phonemes, word)
    decoded_phonemes_list.append(decoded_phonemes)
    decoder_attentions_list.append(decoder_attns)
  return decoded_phonemes_list, decoder_attentions_list

def post_process(
      phonemes:list,
      word:str,
      max_single_repeat:int = 3,
      max_cycle_repeats:int = 2
    ) -> List[str] :
  """
  Post-processes raw phoneme predictions to fix common infinite loop patterns and unnatural repetitions.
  Handles three specific failure cases from sequence-to-sequence G2P models:
  1. Single phonemes stuttering (e.g., ['P', 'P', 'P', ...])
  2. Cyclic pattern repetition (e.g., ['AO', 'W', 'AO', 'W', ...])
  3. Overly long outputs without natural stopping points

  The function guarantees the output will:
  - End with exactly one <EOS> token
  - Have no more than max_single_repeat consecutive identical phonemes
  - Contain no more than max_cycle_repeats instances of any cyclic pattern
  - Be length-constrained relative to input word length (set to length of the word + 1 by default)

  Args:
    phonemes: List of predicted phonemes (may include <EOS>)
    word: Original input word for length reference
    max_single_repeat: Max allowed consecutive repeats of single phoneme
    max_cycle_repeats: Max allowed repeats of any cycle pattern
  Returns:
    Cleaned phoneme list with smart truncation
  """
  # Case 0: Remove any EOS and re-add later
  if "<EOS>" in phonemes :
    phonemes.remove("<EOS>")

  # Case 1: Single phoneme infinite repeat
  for i in range(1, len(phonemes)) :
    if phonemes[i] == phonemes[i-1] :
      repeat_count = 1
      while i+repeat_count < len(phonemes) and phonemes[i+repeat_count] == phonemes[i] :
        repeat_count += 1
      if repeat_count >= max_single_repeat :
        return phonemes[:i] + ["<EOS>"]

  # Case 2: Cycle pattern detection
  for cycle_length in range(1, len(phonemes)//2 + 1) :
    if len(phonemes) < cycle_length * max_cycle_repeats :
      continue

    last_segment = phonemes[-cycle_length*max_cycle_repeats:]
    cycles = [last_segment[i*cycle_length:(i+1)*cycle_length] for i in range(max_cycle_repeats)]

    if all(c == cycles[0] for c in cycles) :
      return phonemes[:-cycle_length*(max_cycle_repeats-1)] + ["<EOS>"]

  # Case 3: Length-based truncation
  max_reasonable_length = len(word)+1
  if len(phonemes) > max_reasonable_length :
    # Find last vowel as natural break point
    vowels = {
      "AA", "AE", "AH", "AO", "AW", "AY",
      "EH", "ER", "EY",
      "IH", "IY",
      "OW", "OY",
      "UH", "UW"
    }
    last_vowel_pos = max([i for i, phoneme in enumerate(phonemes) if phoneme in vowels], default=-1)

    if last_vowel_pos != -1 and last_vowel_pos < len(phonemes)-2 :
      return phonemes[:last_vowel_pos+2] + ["<EOS>"]
    return phonemes[:max_reasonable_length] + ["<EOS>"]

  # Default: Add EOS if missing
  return phonemes + ["<EOS>"]

if __name__ == "__main__" :
  parser = ArgumentParser()
  parser.add_argument("--lang", help="g2p's language model (en_id, en, id)")
  parser.add_argument("--mdl_prefix", help="model prefix", nargs='?', const=1, default='')
  parser.add_argument("--grp_type", help="grapheme type (unigram, bigram, trigram)")
  parser.add_argument("--weight_decay", help="weight decay (1e_5, 1e_4)", nargs='?', const=1, default="1e_5")
  parser.add_argument("--attn_model", help="attention type (dot, general, concat)", nargs='?', const=1, default="dot")
  parser.add_argument("--emb_dim", help="embedding dimension", type=int)
  parser.add_argument("--hidden_size", help="hidden layer size", type=int)
  parser.add_argument("--n_layers", help="number of hidden layers", type=int)
  args = parser.parse_args()

  load_mappings(config=args)
  load_models(config=args)

  while True :
    inp = input("\nInput (enter 'exit' to quit): ").strip()
    if inp.lower() == "exit" :
      confirm = input("Are you sure? (y/n): ").strip().lower()
      if confirm == 'y' :
        break
      else :
        continue
    # Run prediction
    phonemes, attentions = infer_sentence(inp)
    print(phonemes)
