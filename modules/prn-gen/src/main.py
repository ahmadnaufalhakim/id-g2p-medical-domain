from argparse import (
  ArgumentParser,
  Namespace
)
import os
import torch

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

def load_models(config:Namespace) :
  print("Loading models..")
  global encoder, decoder

  suffix_pattern = f"attn_{config.attn_model}-emb_{config.emb_dim}-hddn_{config.hidden_size}-layers_{config.n_layers}-epoch_"
  encoders = sorted([
    f for f in os.listdir(MODELS_DIR) if f.startswith(f"encoder-{suffix_pattern}")],
    key=lambda fn: int(fn.split("epoch_")[1].split(".pth")[0])
  )
  decoders = sorted([
    f for f in os.listdir(MODELS_DIR) if f.startswith(f"decoder-{suffix_pattern}")],
    key=lambda fn: int(fn.split("epoch_")[1].split(".pth")[0])
  )
  assert len(encoders)>0 and len(decoders)>0, \
    f"""No encoder and decoder found with the following parameters:
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

def infer_sentence(sentence:str) :
  words = sentence.split()
  decoded_phonemes_list = []
  decoder_attentions_list = []
  for word in words :
    decoded_phonemes, decoder_attns = infer(word)
    decoded_phonemes_list.append(decoded_phonemes)
    decoder_attentions_list.append(decoder_attns)
  return decoded_phonemes_list, decoder_attentions_list

if __name__ == "__main__" :
  parser = ArgumentParser()
  parser.add_argument("--lang", help="g2p's language model (en_id, en, id)")
  parser.add_argument("--grp_type", help="grapheme type (unigram, bigram, trigram)")
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
