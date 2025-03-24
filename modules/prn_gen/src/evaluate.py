from argparse import (
  ArgumentParser,
  Namespace
)
from jiwer import wer
import os
import torch

from model import (
  DEVICE,
  Encoder, Decoder
)

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(CURR_DIR, "..", "exp")
DATA_DIR = ''
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

val_pairs = []

def load_val_pairs(lang:str) :
  global DATA_DIR
  global val_pairs

  with open(os.path.join(EXPERIMENTS_DIR, f"{lang}/data/val.csv")) as f_csv :
    next(f_csv, None)
    val_pairs = [[s.strip('\n') for s in row.split(',')] for row in f_csv]

def load_mappings(config:Namespace) :
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

def load_models(config:Namespace, mdl_prefix:str = '') :
  print("Loading models..")
  global encoder, decoder

  mdl_suffix = f"-{f'wdecay_{config.weight_decay}-' if hasattr(config, 'weight_decay') and config.weight_decay else ''}attn_{config.attn_model}-emb_{config.emb_dim}-hddn_{config.hidden_size}-layers_{config.n_layers}-epoch_"
  encoders = sorted([
    f for f in os.listdir(MODELS_DIR) if f.startswith(f"{mdl_prefix}encoder{mdl_suffix}")],
    key=lambda fn: int(fn.split("epoch_")[1].split(".pth")[0])
  )
  decoders = sorted([
    f for f in os.listdir(MODELS_DIR) if f.startswith(f"{mdl_prefix}decoder{mdl_suffix}")],
    key=lambda fn: int(fn.split("epoch_")[1].split(".pth")[0])
  )
  assert len(encoders)>0 and len(decoders)>0, \
    f"""No encoder and decoder found with the following parameters:
    - language: {config.lang}
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

def word_to_tensor(word:str) :
  assert GRP_TYPE in ["unigram", "bigram", "trigram"]
  if GRP_TYPE == "unigram" :
    graphemes = [*word]
  elif GRP_TYPE == "bigram" :
    graphemes = [word[i:i+2] for i in range(len(word)-1)] if len(word)>=2 else [word]
  elif GRP_TYPE == "trigram" :
    graphemes = [word[i:i+3] for i in range(len(word)-2)] if len(word)>=3 else [word]

  indexes = [GRAPHEME2INDEX.get(grapheme, UNK_TOKEN) for grapheme in graphemes] + [EOS_TOKEN]
  tensor = torch.LongTensor(indexes).view(-1, 1).to(DEVICE)
  return tensor

def infer(word:str, with_attention:bool=False) :
  input_tensor = word_to_tensor(word)

  # Run through encoder
  encoder_hidden = encoder.init_hidden()
  encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
  # Prepare decoder inputs
  decoder_input = torch.LongTensor([[SOS_TOKEN]]).to(DEVICE)
  decoder_context = torch.zeros(1, decoder.hidden_size).to(DEVICE)
  decoder_hidden = encoder_hidden

  decoder_attentions = torch.zeros(MAX_LENGTH, MAX_LENGTH)
  decoded_phonemes = []
  for di in range(MAX_LENGTH) :
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

def evaluate(lang:str) :
  total_per = .0
  for pair in val_pairs :
    if lang == "en_id" :
      word, arpabet_phoneme_sequence, _ = pair
    elif lang == "en" :
      word, arpabet_phoneme_sequence = pair
    elif lang == "id" :
      word, _, _, arpabet_phoneme_sequence = pair
    output_phonemes, _ = infer(word.lower())
    try :
      output_phonemes.remove("<EOS>")
    except ValueError as e :
      pass
    total_per += wer(
      arpabet_phoneme_sequence,
      ' '.join(output_phonemes)
    )
  return total_per*100/len(val_pairs)

if __name__ == "__main__" :
  langs = ["en_id", "en", "id"]
  grp_types = ["trigram", "bigram"]
  weight_decays = ["1e_5"]
  attn_models = ["dot"]
  emb_dims_list = [[512, 64], [64]]
  hidden_sizes = [64, 50, 32]
  n_layers_list = [1]
  mdl_prefix = "FIN-"

  # Constants (if needed)
  LANG = "en_id"

  # Evaluate
  for lang in langs :
    # load_val_pairs(lang=lang)
    load_val_pairs(lang=LANG)
    for grp_type, emb_dims in zip(grp_types, emb_dims_list) :
      for weight_decay in weight_decays :
        for attn_model in attn_models :
          for emb_dim in emb_dims :
            for hidden_size in hidden_sizes :
              for n_layers in n_layers_list :
                args = Namespace(
                  lang=lang,
                  grp_type=grp_type,
                  weight_decay=weight_decay,
                  attn_model=attn_model,
                  emb_dim=emb_dim,
                  hidden_size=hidden_size,
                  n_layers=n_layers
                )
                load_mappings(config=args)
                try :
                  load_models(config=args, mdl_prefix=mdl_prefix)
                  # print(f"{args}\n{str(evaluate(lang=lang)).replace('.',',')}\n\n")
                  print(f"{args}\n{str(evaluate(lang=LANG)).replace('.',',')}\n\n")
                except AssertionError as e :
                  print(f"model with parameters {args} not found.\nskipping..\n\n")