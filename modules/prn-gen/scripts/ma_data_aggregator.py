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

# phoneme_to_number_of_phoneme_occurences = {phn:0 for phoneme in [PHONEMES, D_PHONEMES, S_PHONEMES, T_PHONEMES] for phn in phoneme}
# with open(os.path.join(DATA_DIR, "ma/train.csv")) as f_read, open(os.path.join(DATA_DIR, "ma/train_converted.csv"), 'w') as f_write :
#   csv_reader = csv.reader(f_read)
#   csv_writer = csv.writer(f_write)
#   headers = next(csv_reader, None)
#   if headers :
#     csv_writer.writerow(headers)
#   for row in csv_reader :
#     # Count phoneme distribution
#     for phoneme in PHONEMES :
#       phoneme_to_number_of_phoneme_occurences[phoneme] += row[1].count(phoneme)
#     ## d IPA phonemes
#     if D_PHONEMES[0] in row[1] :
#       phoneme_to_number_of_phoneme_occurences[D_PHONEMES[0]] += row[1].count(D_PHONEMES[0])
#     else :
#       phoneme_to_number_of_phoneme_occurences[D_PHONEMES[1]] += row[1].count(D_PHONEMES[1])
#     ## s IPA phonemes
#     if S_PHONEMES[0] in row[1] :
#       phoneme_to_number_of_phoneme_occurences[S_PHONEMES[0]] += row[1].count(S_PHONEMES[0])
#     else :
#       phoneme_to_number_of_phoneme_occurences[S_PHONEMES[1]] += row[1].count(S_PHONEMES[1])
#     ## t IPA phonemes
#     if T_PHONEMES[0] in row[1] :
#       phoneme_to_number_of_phoneme_occurences[T_PHONEMES[0]] += row[1].count(T_PHONEMES[0])
#     else :
#       phoneme_to_number_of_phoneme_occurences[T_PHONEMES[1]] += row[1].count(T_PHONEMES[1])

#     # Rewrite phonemes as a space-separated phoneme sequence
#     phoneme_sequence = []
#     row[1] = row[1].replace('-', '')
#     for i in range(len(row[1])) :
#       if row[1][i] in PHONEMES :
#         phoneme_sequence.append(IPA_TO_2_LETTER_ARPABET[row[1][i]])
#       ## Handle double letter IPA phoneme
#       elif row[1][i:i+2] in D_PHONEMES :
#         phoneme_sequence.append(D_IPA_TO_2_LETTER_ARPABET[row[1][i:i+2]])
#         i += 1
#       elif row[1][i:i+2] in S_PHONEMES :
#         phoneme_sequence.append(S_IPA_TO_2_LETTER_ARPABET[row[1][i:i+2]])
#         i += 1
#       elif row[1][i:i+2] in T_PHONEMES :
#         phoneme_sequence.append(T_IPA_TO_2_LETTER_ARPABET[row[1][i:i+2]])
#         i += 1
#       ## Handle rest of single letter IPA phoneme
#       elif row[1][i] in D_PHONEMES :
#         phoneme_sequence.append(D_IPA_TO_2_LETTER_ARPABET[row[1][i]])
#       elif row[1][i] in S_PHONEMES :
#         phoneme_sequence.append(S_IPA_TO_2_LETTER_ARPABET[row[1][i]])
#       elif row[1][i] in T_PHONEMES :
#         phoneme_sequence.append(T_IPA_TO_2_LETTER_ARPABET[row[1][i]])
#     csv_writer.writerow([row[0],' '.join(phoneme_sequence)])
# print(len(phoneme_to_number_of_phoneme_occurences), phoneme_to_number_of_phoneme_occurences)

# with open(os.path.join(DATA_DIR, "ma/train.csv")) as f_read, open(os.path.join(DATA_DIR, "ma/train_converted.csv"), 'w') as f_write :
#   csv_reader = csv.reader(f_read)
#   csv_writer = csv.writer(f_write)
#   headers = next(csv_reader, None)
#   if headers :
#     csv_writer.writerow([headers[1]])
#   for row in csv_reader :
#     csv_writer.writerow([row[1]])