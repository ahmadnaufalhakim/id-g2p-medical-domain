import csv
import os
import re

# Constants
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "..", "data")
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
  's': 'S',
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
PHONEMES = [
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
D_PHONEMES = [
  "dʒ", # j
  'd'
]
S_PHONEMES = [
  "sj", # sh,sy
  's'
]
T_PHONEMES = [
  "tʃ", # c
  't'
]

rows = []
train_csv_filenames = [f for f in os.listdir(os.path.join(DATA_DIR, "ma")) if re.match(r"train_(\d+)-(\d+)_(\d+)-(\d+)-(\d+)_(\d+)-(\d+)-(\d+).csv", f)]
for train_csv_filename in train_csv_filenames :
  with open(os.path.join(DATA_DIR, "ma", train_csv_filename)) as f_read :
    for line in f_read :
      rows.append(line.strip())
rows = sorted(rows, key=lambda row: int(row.split(',')[0]))
with open(os.path.join(DATA_DIR, "ma", "train.csv"), 'w') as f_write :
  f_write.write("word,phoneme,word_syllable_sequence,phoneme_syllable_sequence\n")
  for row in rows :
    f_write.write(f"{','.join(row.split(',')[1:])}\n")
