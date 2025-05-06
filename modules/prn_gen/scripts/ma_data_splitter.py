"""
To split data from data/ma/train.csv to each respective train val test sets
"""

import csv
import os
import random

random.seed(23522026)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

IPA_TO_2_LETTER_ARPABET = {
  'ʔ': 'Q',
  'b': 'B',
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
  'p': 'P',
  'r': 'R',
  'u': "UW",
  'v': 'V',
  'w': 'W',
  'j': 'Y',  # y
  'z': 'Z',
}
A_IPA_TO_2_LETTER_ARPABET = {
  "ai": "AY", # diftong ai
  "au": "AW", # diftong au
  'a': 'AA'
}
D_IPA_TO_2_LETTER_ARPABET = {
  "dʒ": "JH", # j
  'd': 'D'
}
E_IPA_TO_2_LETTER_ARPABET = {
  "ei": "EY", # diftong ei
  'e': "EH"
}
O_IPA_TO_2_LETTER_ARPABET = {
  "oi": "OY", # diftong oi
  'o': "AO"
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
  'b',
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
  'p',
  'r',
  'u',
  'v',
  'w',
  'j',  # y
  'z'
]
A_PHONEMES = [
  "ai", # diftong ai
  "au", # diftong au
  'a'
]
D_PHONEMES = [
  "dʒ", # j
  'd'
]
E_PHONEMES = [
  "ei", # diftong ei
  'e'
]
O_PHONEMES = [
  "oi", # diftong oi
  'o'
]
S_PHONEMES = [
  "sj", # sh,sy
  's'
]
T_PHONEMES = [
  "tʃ", # c
  't'
]

# Accumulate all words for each phoneme
phoneme_to_words = {
  phn: {
    "words": set()
  } for phoneme in [PHONEMES, A_PHONEMES, D_PHONEMES, E_PHONEMES, O_PHONEMES, S_PHONEMES, T_PHONEMES] for phn in phoneme
}
with open(os.path.join(DATA_DIR, "ma/train.csv")) as f_read :
  csv_reader = csv.reader(f_read)
  headers = next(csv_reader, None)
  for row in csv_reader :
    for syllable in row[3].split('.') :
      # Count phoneme distribution
      for phoneme in PHONEMES :
        if phoneme in syllable :
          phoneme_to_words[phoneme]["words"].add(tuple(row))
      ## a IPA phonemes
      if A_PHONEMES[0] in syllable :
        phoneme_to_words[A_PHONEMES[0]]["words"].add(tuple(row))
      elif A_PHONEMES[1] in syllable :
        phoneme_to_words[A_PHONEMES[1]]["words"].add(tuple(row))
      elif A_PHONEMES[2] in syllable :
        phoneme_to_words[A_PHONEMES[2]]["words"].add(tuple(row))
      ## d IPA phonemes
      if D_PHONEMES[0] in syllable :
        phoneme_to_words[D_PHONEMES[0]]["words"].add(tuple(row))
      elif D_PHONEMES[1] in syllable :
        phoneme_to_words[D_PHONEMES[1]]["words"].add(tuple(row))
      ## e IPA phonemes
      if E_PHONEMES[0] in syllable :
        phoneme_to_words[E_PHONEMES[0]]["words"].add(tuple(row))
      elif E_PHONEMES[1] in syllable :
        phoneme_to_words[E_PHONEMES[1]]["words"].add(tuple(row))
      ## o IPA phonemes
      if O_PHONEMES[0] in syllable :
        phoneme_to_words[O_PHONEMES[0]]["words"].add(tuple(row))
      elif O_PHONEMES[1] in syllable :
        phoneme_to_words[O_PHONEMES[1]]["words"].add(tuple(row))
      ## s IPA phonemes
      if S_PHONEMES[0] in syllable :
        phoneme_to_words[S_PHONEMES[0]]["words"].add(tuple(row))
      elif S_PHONEMES[1] in syllable :
        phoneme_to_words[S_PHONEMES[1]]["words"].add(tuple(row))
      ## t IPA phonemes
      if T_PHONEMES[0] in syllable :
        phoneme_to_words[T_PHONEMES[0]]["words"].add(tuple(row))
      elif T_PHONEMES[1] in syllable :
        phoneme_to_words[T_PHONEMES[1]]["words"].add(tuple(row))

# Sort phoneme by phoneme occurrences (ascending)
phoneme_to_words = dict(sorted(phoneme_to_words.items(), key=lambda item: len(item[1]["words"])))
# Convert phoneme word set to list
for k, v in phoneme_to_words.items() :
  phoneme_to_words[k] = {
    "words": sorted(list(v["words"]), key=lambda entry: entry[0])
  }

# Start of the TRAIN/VAL/TEST set splitting
train, val, test = set(), set(), set()
train_percentage = .9
phone_keys = list(phoneme_to_words.keys())
for i in range(len(phone_keys)) :
  phn = phone_keys[i]
  ## Populate the TEST set
  ### Shuffle the words list
  random.shuffle(phoneme_to_words[phn]["words"])
  ### Split the words list
  split_index = round(train_percentage * len(phoneme_to_words[phn]["words"]))
  test_split = set(phoneme_to_words[phn]["words"][split_index:])
  ### Assign words to TEST set, ensuring no duplicates
  test.update(test_split)
  ### Remove assigned words from other phonemes' word lists
  for other_phn, other_data in phoneme_to_words.items() :
    phoneme_to_words[other_phn]["words"] = list(set(other_data["words"]) - test_split)

  ## Populate TRAIN and VAL set
  ### Shuffle the words list
  random.shuffle(phoneme_to_words[phn]["words"])
  ### Split the words list
  split_index = round(train_percentage * len(phoneme_to_words[phn]["words"]))
  train_split = set(phoneme_to_words[phn]["words"][:split_index])
  val_split = set(phoneme_to_words[phn]["words"][split_index:])
  ### Assign words to TRAIN and VAL set, ensuring no duplicates
  train.update(train_split)
  val.update(val_split)
  ### Remove assigned words from other phonemes' word lists
  for other_phn, other_data in phoneme_to_words.items() :
    phoneme_to_words[other_phn]["words"] = list(set(other_data["words"]) - train_split - val_split)
  ### Re-sort the phones based on their occurrences
  phoneme_to_words = dict(sorted(phoneme_to_words.items(), key=lambda item: len(item[1]["words"])))
  phone_keys = list(phoneme_to_words.keys())
# End of the TRAIN/VAL/TEST set splitting

def convert_to_arpabet(syllables:list) :
  result = []
  for syllable in syllables :
    i = 0
    while i<len(syllable) :
      # Handle 2-letter IPA phoneme
      if syllable[i:i+2] in A_PHONEMES :
        result.append(A_IPA_TO_2_LETTER_ARPABET[syllable[i:i+2]])
        i += 2
      elif syllable[i:i+2] in D_PHONEMES :
        result.append(D_IPA_TO_2_LETTER_ARPABET[syllable[i:i+2]])
        i += 2
      elif syllable[i:i+2] in E_PHONEMES :
        result.append(E_IPA_TO_2_LETTER_ARPABET[syllable[i:i+2]])
        i += 2
      elif syllable[i:i+2] in O_PHONEMES :
        result.append(O_IPA_TO_2_LETTER_ARPABET[syllable[i:i+2]])
        i += 2
      elif syllable[i:i+2] in S_PHONEMES :
        result.append(S_IPA_TO_2_LETTER_ARPABET[syllable[i:i+2]])
        i += 2
      elif syllable[i:i+2] in T_PHONEMES :
        result.append(T_IPA_TO_2_LETTER_ARPABET[syllable[i:i+2]])
        i += 2
      ## Handle rest of single letter IPA phoneme
      elif syllable[i] in A_PHONEMES :
        result.append(A_IPA_TO_2_LETTER_ARPABET[syllable[i]])
        i += 1
      elif syllable[i] in D_PHONEMES :
        result.append(D_IPA_TO_2_LETTER_ARPABET[syllable[i]])
        i += 1
      elif syllable[i] in E_PHONEMES :
        result.append(E_IPA_TO_2_LETTER_ARPABET[syllable[i]])
        i += 1
      elif syllable[i] in O_PHONEMES :
        result.append(O_IPA_TO_2_LETTER_ARPABET[syllable[i]])
        i += 1
      elif syllable[i] in S_PHONEMES :
        result.append(S_IPA_TO_2_LETTER_ARPABET[syllable[i]])
        i += 1
      elif syllable[i] in T_PHONEMES :
        result.append(T_IPA_TO_2_LETTER_ARPABET[syllable[i]])
        i += 1
      elif syllable[i] in PHONEMES :
        result.append(IPA_TO_2_LETTER_ARPABET[syllable[i]])
        i += 1
      else :
        i += 1
  return result

n_train = len(list(train))
n_val = len(list(val))
n_test = len(list(test))
with open(os.path.join(DATA_DIR, "ma/train.csv")) as f_read,\
     open(os.path.join(DATA_DIR, "ma/train_converted.csv"), 'w') as f_train_write,\
     open(os.path.join(DATA_DIR, "ma/val_converted.csv"), 'w') as f_val_write,\
     open(os.path.join(DATA_DIR, "ma/test_converted.csv"), 'w') as f_test_write :
  csv_reader = csv.reader(f_read)
  train_csv_writer = csv.writer(f_train_write)
  val_csv_writer = csv.writer(f_val_write)
  test_csv_writer = csv.writer(f_test_write)

  headers = next(csv_reader, None)
  del headers[1]; headers[2] = f"ipa_{headers[2]}"
  if headers :
    train_csv_writer.writerow(headers+["arpabet_phoneme_sequence"])
    val_csv_writer.writerow(headers+["arpabet_phoneme_sequence"])
    test_csv_writer.writerow(headers+["arpabet_phoneme_sequence"])
  print(f"Populating train set .. ({n_train})")
  for train_entry in sorted(list(train), key=lambda entry: entry[0]) :
    phoneme_sequence = convert_to_arpabet(syllables=train_entry[3].split('.'))
    train_csv_writer.writerow([train_entry[0], *train_entry[2:4], ' '.join(phoneme_sequence)])
  print("Done populating train set")
  print(f"Populating val set .. ({n_val})")
  for val_entry in sorted(list(val), key=lambda entry: entry[0]) :
    phoneme_sequence = convert_to_arpabet(syllables=val_entry[3].split('.'))
    val_csv_writer.writerow([val_entry[0], *val_entry[2:4], ' '.join(phoneme_sequence)])
  print("Done populating val set")
  print(f"Populating test set .. ({n_test})")
  for test_entry in sorted(list(test), key=lambda entry: entry[0]) :
    phoneme_sequence = convert_to_arpabet(syllables=test_entry[3].split('.'))
    test_csv_writer.writerow([test_entry[0], *test_entry[2:4], ' '.join(phoneme_sequence)])
  print("Done populating test set")
  print(f"|9-(train/val)| + |8.1-(train/test)| = {abs(9-(n_train/n_val)) + abs(8.1-(n_train/n_test))}")
