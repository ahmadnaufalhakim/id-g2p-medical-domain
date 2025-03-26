from argparse import Namespace
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = (DEVICE.type == "cuda")

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(CURR_DIR, "..", "exp")
MODELS_DIR = ''

SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2
UNK_TOKEN = 3

"""Encoder definition"""
class Encoder(nn.Module) :
  def __init__(self, input_size, emb_dim, hidden_size, n_layers=1) -> None :
    super(Encoder, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.n_layers = n_layers

    self.embedding = nn.Embedding(input_size, emb_dim)
    self.gru = nn.GRU(emb_dim, hidden_size, n_layers, batch_first=False)
    if USE_CUDA :
      self.embedding = self.embedding.cuda()
      self.gru = self.gru.cuda()

  def forward(self, token_inputs, hidden) :
    embedded = self.embedding(token_inputs) # [seq_len, batch_size, emb_dim]
    output, hidden = self.gru(embedded, hidden)
    return output, hidden # output: [seq_len, batch_size, hidden_size]

  def init_hidden(self, batch_size=1) :
    hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
    # hidden shape: [n_layers, batch_size, hidden_size]
    if USE_CUDA :
      hidden = hidden.cuda()
    return hidden

"""Attention definition"""
class Attn(nn.Module) :
  def __init__(self, method, hidden_size) -> None :
    super(Attn, self).__init__()
    self.method = method
    self.hidden_size = hidden_size

    if self.method == "general" :
      self.attn = nn.Linear(self.hidden_size, hidden_size)
      if USE_CUDA :
        self.attn = self.attn.cuda()
    elif self.method == "concat" :
      self.attn = nn.Linear(self.hidden_size*2, hidden_size)
      self.v = nn.Parameter(torch.FloatTensor(hidden_size))
      if USE_CUDA :
        self.attn = self.attn.cuda()
        self.v = self.v.cuda()

  def forward(self, hidden, encoder_outputs) :
    # hidden shape: [1, batch_size, hidden_size]
    # encoder_outputs shape: [seq_len, batch_size, hidden_size]

    if self.method == "dot" :
      # Vectorized dot product for all positions in the sequence
      attn_energies = torch.sum(hidden * encoder_outputs, dim=2) # [seq_len, batch_size]
    elif self.method == "general" :
      energy = self.attn(encoder_outputs) # [seq_len, batch_size, hidden_size]
      attn_energies = torch.sum(hidden * energy, dim=2)
    elif self.method == "concat" :
      hidden_expanded = hidden.expand(encoder_outputs.size(0), -1, -1) # [seq_len, batch_size, hidden_size]
      energy = self.attn(torch.cat((hidden_expanded, encoder_outputs), 2)) # [seq_len, batch_size, hidden_size]
      attn_energies = torch.sum(self.v * energy, dim=2)

    # Normalize energies to weights
    attn_weights = F.softmax(attn_energies, dim=0) # [seq_len, batch_size]
    return attn_weights.transpose(0, 1).unsqueeze(1) # [batch_size, 1, seq_len]

"""Decoder definition"""
class Decoder(nn.Module) :
  def __init__(self, attn_model, emb_dim, hidden_size, output_size, n_layers=1, dropout_proba=.1) -> None :
    super(Decoder, self).__init__()
    self.attn_model = attn_model
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = n_layers
    self.dropout_proba = dropout_proba

    # Define layers
    self.embedding = nn.Embedding(output_size, emb_dim)
    self.gru = nn.GRU(emb_dim + hidden_size, hidden_size, n_layers, dropout=dropout_proba, batch_first=False)
    self.out = nn.Linear(hidden_size*2, output_size)

    # Choose attention model
    if attn_model != "none" :
      self.attn = Attn(attn_model, hidden_size)

    if USE_CUDA :
      self.embedding = self.embedding.cuda()
      self.gru = self.gru.cuda()
      self.out = self.out.cuda()
      self.attn = self.attn.cuda()

  def forward(self, token_input, last_context, last_hidden, encoder_outputs) :
    # token_input shape: [1, batch_size]
    # last_context shape: [batch_size, hidden_size]
    # last_hidden shape: [n_layers, batch_size, hidden_size]
    # encoder_outputs shape: [seq_len, batch_size, hidden]
    # Get the embedding of the current input token (last output token)

    embedded = self.embedding(token_input) # [1, batch_size, emb_dim]
    # Combine embedded input token and last context, run through RNN
    rnn_input = torch.cat((embedded, last_context.unsqueeze(0)), dim=2) # [1, batch_size, emb_dim + hidden_size]
    # GRU forward
    rnn_output, hidden = self.gru(rnn_input, last_hidden) # rnn_output: [1, batch_size, hidden_size]

    # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
    attn_weights = self.attn(rnn_output, encoder_outputs) # [batch_size, 1, seq_len]
    context = torch.bmm(attn_weights, encoder_outputs.transpose(0, 1)) # [batch_size, 1, hidden_size]
    context = context.transpose(0, 1) # [1, batch_size, hidden_size]

    # Final output layer (next token prediction) using the RNN hidden state and context vector
    rnn_output = rnn_output.squeeze(0)  # [batch_size, hidden_size]
    context = context.squeeze(0)        # [batch_size, hidden_size]
    output = torch.cat((rnn_output, context), dim=1) # [batch_size, hidden_size * 2]
    output = F.log_softmax(self.out(output), dim=1) # [batch_size, output_size]

    # Return final output, hidden state, and attention weights (for visualization)
    return output, context, hidden, attn_weights

class G2P :
  def __init__(self, config:Namespace) -> None :
    self.GRP_TYPE = ''
    self.MAX_LENGTH = 32
    self.INDEX2GRAPHEME = None
    self.GRAPHEME2INDEX = None
    self.INDEX2PHONEME = None
    self.encoder = None
    self.decoder = None

    global MODELS_DIR
    MODELS_DIR = os.path.join(EXPERIMENTS_DIR, f"{config.lang}/models/{config.grp_type}")
    self.load_mappings(config=config)
    self.load_models(config=config)

  def load_mappings(self, config:Namespace) -> None :
    print("Loading mappings..")
    if not os.path.exists(MODELS_DIR) :
      raise Exception("models directory not found")
    self.GRP_TYPE = config.grp_type
    print("\tloading id2grp")
    self.INDEX2GRAPHEME = torch.load(os.path.join(MODELS_DIR, "id2grp.pth"), weights_only=True)
    self.INDEX2GRAPHEME = {
      k: v.lower() if k not in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN] else v for k, v in self.INDEX2GRAPHEME.items()
    }
    self.GRAPHEME2INDEX = {
      v: k for k, v in self.INDEX2GRAPHEME.items() if k not in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN]
    }
    print("\tloading id2phn")
    self.INDEX2PHONEME = torch.load(os.path.join(MODELS_DIR, "id2phn.pth"), weights_only=True)

  def load_models(self, config:Namespace) :
    print("Loading models..")
    mdl_prefix = f"{config.mdl_prefix}-" if hasattr(config, "mdl_prefix") and config.mdl_prefix else ''
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
    self.encoder = Encoder(len(self.INDEX2GRAPHEME), config.emb_dim, config.hidden_size, config.n_layers)
    self.decoder = Decoder(config.attn_model, config.emb_dim, config.hidden_size, len(self.INDEX2PHONEME), config.n_layers)
    # Load weights
    print("\tloading encoder's state dict")
    self.encoder.load_state_dict(torch.load(os.path.join(MODELS_DIR, best_encoder), map_location=DEVICE, weights_only=True))
    print("\tencoder's state dict loaded")
    print("\tloading decoder's state dict")
    self.decoder.load_state_dict(torch.load(os.path.join(MODELS_DIR, best_decoder), map_location=DEVICE, weights_only=True))
    print("\tdecoder's state dict loaded")
    # Move to GPU if available
    self.encoder = self.encoder.to(DEVICE).eval()
    self.decoder = self.decoder.to(DEVICE).eval()

  def word_to_tensor(self, word:str) -> torch.Tensor :
    assert self.GRP_TYPE in ["unigram", "bigram", "trigram"]
    if self.GRP_TYPE == "unigram" :
      graphemes = [*word]
    elif self.GRP_TYPE == "bigram" :
      graphemes = [word[i:i+2] for i in range(len(word)-1)] if len(word)>=2 else [word]
    elif self.GRP_TYPE == "trigram" :
      graphemes = [word[i:i+3] for i in range(len(word)-2)] if len(word)>=3 else [word]
    indexes = [self.GRAPHEME2INDEX.get(grapheme, UNK_TOKEN) for grapheme in graphemes] + [EOS_TOKEN]
    tensor = torch.LongTensor(indexes).view(-1, 1).to(DEVICE)
    return tensor

  def infer(self, word:str, with_attention:bool = False) -> Tuple[list, torch.Tensor] :
    input_tensor = self.word_to_tensor(word)
    # Run through encoder
    encoder_hidden = self.encoder.init_hidden()
    encoder_outputs, encoder_hidden = self.encoder(input_tensor, encoder_hidden)
    # Prepare decoder inputs
    decoder_input = torch.LongTensor([[SOS_TOKEN]]).to(DEVICE)
    decoder_context = torch.zeros(1, self.decoder.hidden_size).to(DEVICE)
    decoder_hidden = encoder_hidden
    decoder_attentions = torch.zeros(self.MAX_LENGTH, self.MAX_LENGTH)
    decoded_phonemes = []
    for di in range(self.MAX_LENGTH) :
      decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(
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
        decoded_phonemes.append(self.INDEX2PHONEME[ni.item()])
      # Next decoder input is last token
      decoder_input = torch.LongTensor([[ni.item()]]).to(DEVICE)
    return decoded_phonemes, decoder_attentions[:di+1, 1:len(encoder_outputs)]

  def __call__(self, input:str) -> Tuple[list, list] :
    words = input.split()
    decoded_phonemes_list = []
    decoder_attentions_list = []
    for word in words :
      decoded_phonemes, decoder_attns = self.infer(word.lower())
      decoded_phonemes_list.append(decoded_phonemes)
      decoder_attentions_list.append(decoder_attns)
    return decoded_phonemes_list, decoder_attentions_list
