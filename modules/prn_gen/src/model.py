import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = (DEVICE.type == "cuda")

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