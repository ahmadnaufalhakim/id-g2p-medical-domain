import csv
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

IPA_TO_2_LETTER_ARPABET = {
  'ʔ': 'Q',
  'a': "AA",
  'b': 'B',
  'e': "EH",
  'ə': "AX",
  'f': 'F',
  'g': 'G',
  'h': "HH",
  'i': "IY",
  'k': 'K',
  'l': 'L',
  'm': 'M',
  'n': 'N',
  'ŋ': "NG",  # ng
  'ɲ': "NY",  # ny
  'o': "AO",
  'p': 'P',
  'r': 'R',
  'u': "UW",
  'v': 'V',
  'w': 'W',
  'j': 'Y',  # y
  'z': 'Z',
}
D_IPA_TO_2_LETTER_ARPABET = {
  "dʒ": "JH", # j
  'd': 'D'
}
S_IPA_TO_2_LETTER_ARPABET = {
  "sj": "SH", # sh,sy
  's': 'S'
}
T_IPA_TO_2_LETTER_ARPABET = {
  "tʃ": "CH", # c
  't': 'T'
}

phonemes = [
  'ʔ',
  'a',
  'b',
  'e',
  'ə',
  'f',
  'g',
  'h',
  'i',
  'k',
  'l',
  'm',
  'n',
  'ŋ',  # ng
  'ɲ',  # ny
  'o',
  'p',
  'r',
  'u',
  'v',
  'w',
  'j',  # y
  'z'
]
d_phonemes = [
  "dʒ", # j
  'd'
]
s_phonemes = [
  "sj", # sh,sy
  's'
]
t_phonemes = [
  "tʃ", # c
  't'
]

phoneme_to_number_of_phoneme_occurences = {phn:0 for phoneme in [phonemes, d_phonemes, s_phonemes, t_phonemes] for phn in phoneme}
with open(os.path.join(DATA_DIR, "id/train.csv")) as f_read, open(os.path.join(DATA_DIR, "id/train_converted.csv"), 'w') as f_write :
  csv_reader = csv.reader(f_read)
  csv_writer = csv.writer(f_write)
  headers = next(csv_reader, None)
  if headers :
    csv_writer.writerow(headers)
  for row in csv_reader :
    # Count phoneme distribution
    for phoneme in phonemes :
      phoneme_to_number_of_phoneme_occurences[phoneme] += row[1].count(phoneme)
    ## d IPA phonemes
    if d_phonemes[0] in row[1] :
      phoneme_to_number_of_phoneme_occurences[d_phonemes[0]] += row[1].count(d_phonemes[0])
    else :
      phoneme_to_number_of_phoneme_occurences[d_phonemes[1]] += row[1].count(d_phonemes[1])
    ## s IPA phonemes
    if s_phonemes[0] in row[1] :
      phoneme_to_number_of_phoneme_occurences[s_phonemes[0]] += row[1].count(s_phonemes[0])
    else :
      phoneme_to_number_of_phoneme_occurences[s_phonemes[1]] += row[1].count(s_phonemes[1])
    ## t IPA phonemes
    if t_phonemes[0] in row[1] :
      phoneme_to_number_of_phoneme_occurences[t_phonemes[0]] += row[1].count(t_phonemes[0])
    else :
      phoneme_to_number_of_phoneme_occurences[t_phonemes[1]] += row[1].count(t_phonemes[1])

    # Rewrite phonemes as a space-separated phoneme sequence
    phoneme_sequence = []
    row[1] = row[1].replace('-', '')
    for i in range(len(row[1])) :
      if row[1][i] in phonemes :
        phoneme_sequence.append(IPA_TO_2_LETTER_ARPABET[row[1][i]])
      ## Handle double letter IPA phoneme
      elif row[1][i:i+2] in d_phonemes :
        phoneme_sequence.append(D_IPA_TO_2_LETTER_ARPABET[row[1][i:i+2]])
        i += 1
      elif row[1][i:i+2] in s_phonemes :
        phoneme_sequence.append(S_IPA_TO_2_LETTER_ARPABET[row[1][i:i+2]])
        i += 1
      elif row[1][i:i+2] in t_phonemes :
        phoneme_sequence.append(T_IPA_TO_2_LETTER_ARPABET[row[1][i:i+2]])
        i += 1
      ## Handle rest of single letter IPA phoneme
      elif row[1][i] in d_phonemes :
        phoneme_sequence.append(D_IPA_TO_2_LETTER_ARPABET[row[1][i]])
      elif row[1][i] in s_phonemes :
        phoneme_sequence.append(S_IPA_TO_2_LETTER_ARPABET[row[1][i]])
      elif row[1][i] in t_phonemes :
        phoneme_sequence.append(T_IPA_TO_2_LETTER_ARPABET[row[1][i]])
    csv_writer.writerow([row[0],' '.join(phoneme_sequence)])

print(len(phoneme_to_number_of_phoneme_occurences), phoneme_to_number_of_phoneme_occurences)