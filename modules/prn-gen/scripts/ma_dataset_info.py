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

phoneme_to_words = {
  phn: {
    "words": set()
  } for phoneme in [PHONEMES, A_PHONEMES, D_PHONEMES, E_PHONEMES, O_PHONEMES, S_PHONEMES, T_PHONEMES] for phn in phoneme
}
with open(os.path.join(DATA_DIR, "ma/train.csv")) as f_read, open(os.path.join(DATA_DIR, "ma/train_converted.csv"), 'w') as f_write :
  csv_reader = csv.reader(f_read)
  csv_writer = csv.writer(f_write)
  headers = next(csv_reader, None)
  if headers :
    csv_writer.writerow(headers)
  for row in csv_reader :
    phoneme_sequence = []
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

      # Rewrite phonemes as a space-separated phoneme sequence
      row[1] = row[1].replace('-', '')
      for i in range(len(syllable)) :
        if syllable[i] in PHONEMES :
          phoneme_sequence.append(IPA_TO_2_LETTER_ARPABET[syllable[i]])
        ## Handle double letter IPA phoneme
        elif syllable[i:i+2] in A_PHONEMES :
          phoneme_sequence.append(A_IPA_TO_2_LETTER_ARPABET[syllable[i:i+2]])
          i += 1
        elif syllable[i:i+2] in D_PHONEMES :
          phoneme_sequence.append(D_IPA_TO_2_LETTER_ARPABET[syllable[i:i+2]])
          i += 1
        elif syllable[i:i+2] in E_PHONEMES :
          phoneme_sequence.append(E_IPA_TO_2_LETTER_ARPABET[syllable[i:i+2]])
          i += 1
        elif syllable[i:i+2] in O_PHONEMES :
          phoneme_sequence.append(O_IPA_TO_2_LETTER_ARPABET[syllable[i:i+2]])
          i += 1
        elif syllable[i:i+2] in S_PHONEMES :
          phoneme_sequence.append(S_IPA_TO_2_LETTER_ARPABET[syllable[i:i+2]])
          i += 1
        elif syllable[i:i+2] in T_PHONEMES :
          phoneme_sequence.append(T_IPA_TO_2_LETTER_ARPABET[syllable[i:i+2]])
          i += 1
        ## Handle rest of single letter IPA phoneme
        elif syllable[i] in A_PHONEMES :
          phoneme_sequence.append(A_IPA_TO_2_LETTER_ARPABET[syllable[i]])
        elif syllable[i] in D_PHONEMES :
          phoneme_sequence.append(D_IPA_TO_2_LETTER_ARPABET[syllable[i]])
        elif syllable[i] in E_PHONEMES :
          phoneme_sequence.append(E_IPA_TO_2_LETTER_ARPABET[syllable[i]])
        elif syllable[i] in O_PHONEMES :
          phoneme_sequence.append(O_IPA_TO_2_LETTER_ARPABET[syllable[i]])
        elif syllable[i] in S_PHONEMES :
          phoneme_sequence.append(S_IPA_TO_2_LETTER_ARPABET[syllable[i]])
        elif syllable[i] in T_PHONEMES :
          phoneme_sequence.append(T_IPA_TO_2_LETTER_ARPABET[syllable[i]])
    csv_writer.writerow([row[0],' '.join(phoneme_sequence)])

# Sort phoneme by phoneme occurrences (ascending)
phoneme_to_words = dict(sorted(phoneme_to_words.items(), key=lambda item: len(item[1]["words"])))
# Convert phoneme word set to list
for k, v in phoneme_to_words.items() :
  phoneme_to_words[k] = {
    "words": sorted(list(v["words"]), key=lambda entry: entry[0])
  }

print(phoneme_to_words.keys())
for phn, data in phoneme_to_words.items() :
  print(phn, len(data["words"]))

train, val, test = set(), set(), set()
train_percentage = .9
phone_keys = list(phoneme_to_words.keys())
# split TRAIN/VAL/TEST process start
for i in range(len(phone_keys)) :
  # if i == 5 :
  #   break
  phn = phone_keys[i]
  # populate TEST set
  ## shuffle the words list
  random.shuffle(phoneme_to_words[phn]["words"])
  ## split the words list
  split_index = round(train_percentage * len(phoneme_to_words[phn]["words"]))
  test_split = set(phoneme_to_words[phn]["words"][split_index:])
  ## assign words test, ensuring no duplicates
  test.update(test_split)
  ## remove assigned words from other phoneme's word lists
  for other_phn, other_data in phoneme_to_words.items() :
    phoneme_to_words[other_phn]["words"] = list(set(other_data["words"]) - test_split)

  # populate TRAIN and VAL set
  ## shuffle the words list
  random.shuffle(phoneme_to_words[phn]["words"])
  ## split the words list
  split_index = round(train_percentage * len(phoneme_to_words[phn]["words"]))
  train_split = set(phoneme_to_words[phn]["words"][:split_index])
  val_split = set(phoneme_to_words[phn]["words"][split_index:])
  ## assign words to train and val, ensuring no duplicates
  train.update(train_split)
  val.update(val_split)
  print(f"iteration no.{i+1} (phoneme: {phn})")
  print("current phone keys:", phone_keys)
  print(len(list(train)), len(list(val)), len(list(test)))
  print()
  ## remove assigned words from other phoneme's word lists
  for other_phn, other_data in phoneme_to_words.items() :
    phoneme_to_words[other_phn]["words"] = list(set(other_data["words"]) - train_split - val_split)
  ## resort the phones based on their occurrences
  phoneme_to_words = dict(sorted(phoneme_to_words.items(), key=lambda item: len(item[1]["words"])))
  phone_keys = list(phoneme_to_words.keys())

print(phoneme_to_words.keys())
for phn, data in phoneme_to_words.items() :
  print(phn, len(data["words"]))

# split TRAIN/VAL/TEST process end

print(len(train), len(val), len(test))
