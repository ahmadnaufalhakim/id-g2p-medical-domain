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
  's': 'S',
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

phonemes = [
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
a_phonemes = [
  "ai", # diftong ai
  "au", # diftong au
  'a'
]
d_phonemes = [
  "dʒ", # j
  'd'
]
e_phonemes = [
  "ei", # diftong ei
  'e'
]
o_phonemes = [
  "oi", # diftong oi
  'o'
]
s_phonemes = [
  "sj", # sh,sy
  's'
]
t_phonemes = [
  "tʃ", # c
  't'
]

phoneme_to_words = {
  phn: {
    "words": set()
  } for phoneme in [phonemes, a_phonemes, d_phonemes, e_phonemes, o_phonemes, s_phonemes, t_phonemes] for phn in phoneme
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
      for phoneme in phonemes :
        if phoneme in syllable :
          phoneme_to_words[phoneme]["words"].add(tuple(row))
      ## a IPA phonemes
      if a_phonemes[0] in syllable :
        phoneme_to_words[a_phonemes[0]]["words"].add(tuple(row))
      elif a_phonemes[1] in syllable :
        phoneme_to_words[a_phonemes[1]]["words"].add(tuple(row))
      elif a_phonemes[2] in syllable :
        phoneme_to_words[a_phonemes[2]]["words"].add(tuple(row))
      ## d IPA phonemes
      if d_phonemes[0] in syllable :
        phoneme_to_words[d_phonemes[0]]["words"].add(tuple(row))
      elif d_phonemes[1] in syllable :
        phoneme_to_words[d_phonemes[1]]["words"].add(tuple(row))
      ## e IPA phonemes
      if e_phonemes[0] in syllable :
        phoneme_to_words[e_phonemes[0]]["words"].add(tuple(row))
      elif e_phonemes[1] in syllable :
        phoneme_to_words[e_phonemes[1]]["words"].add(tuple(row))
      ## o IPA phonemes
      if o_phonemes[0] in syllable :
        phoneme_to_words[o_phonemes[0]]["words"].add(tuple(row))
      elif o_phonemes[1] in syllable :
        phoneme_to_words[o_phonemes[1]]["words"].add(tuple(row))
      ## s IPA phonemes
      if s_phonemes[0] in syllable :
        phoneme_to_words[s_phonemes[0]]["words"].add(tuple(row))
      elif s_phonemes[1] in syllable :
        phoneme_to_words[s_phonemes[1]]["words"].add(tuple(row))
      ## t IPA phonemes
      if t_phonemes[0] in syllable :
        phoneme_to_words[t_phonemes[0]]["words"].add(tuple(row))
      elif t_phonemes[1] in syllable :
        phoneme_to_words[t_phonemes[1]]["words"].add(tuple(row))

      # Rewrite phonemes as a space-separated phoneme sequence
      row[1] = row[1].replace('-', '')
      for i in range(len(syllable)) :
        if syllable[i] in phonemes :
          phoneme_sequence.append(IPA_TO_2_LETTER_ARPABET[syllable[i]])
        ## Handle double letter IPA phoneme
        elif syllable[i:i+2] in a_phonemes :
          phoneme_sequence.append(A_IPA_TO_2_LETTER_ARPABET[syllable[i:i+2]])
          i += 1
        elif syllable[i:i+2] in d_phonemes :
          phoneme_sequence.append(D_IPA_TO_2_LETTER_ARPABET[syllable[i:i+2]])
          i += 1
        elif syllable[i:i+2] in e_phonemes :
          # print(syllable[i:i+2])
          phoneme_sequence.append(E_IPA_TO_2_LETTER_ARPABET[syllable[i:i+2]])
          i += 1
        elif syllable[i:i+2] in o_phonemes :
          phoneme_sequence.append(O_IPA_TO_2_LETTER_ARPABET[syllable[i:i+2]])
          i += 1
        elif syllable[i:i+2] in s_phonemes :
          phoneme_sequence.append(S_IPA_TO_2_LETTER_ARPABET[syllable[i:i+2]])
          i += 1
        elif syllable[i:i+2] in t_phonemes :
          phoneme_sequence.append(T_IPA_TO_2_LETTER_ARPABET[syllable[i:i+2]])
          i += 1
        ## Handle rest of single letter IPA phoneme
        elif syllable[i] in a_phonemes :
          phoneme_sequence.append(A_IPA_TO_2_LETTER_ARPABET[syllable[i]])
        elif syllable[i] in d_phonemes :
          phoneme_sequence.append(D_IPA_TO_2_LETTER_ARPABET[syllable[i]])
        elif syllable[i] in e_phonemes :
          phoneme_sequence.append(E_IPA_TO_2_LETTER_ARPABET[syllable[i]])
        elif syllable[i] in o_phonemes :
          phoneme_sequence.append(O_IPA_TO_2_LETTER_ARPABET[syllable[i]])
        elif syllable[i] in s_phonemes :
          phoneme_sequence.append(S_IPA_TO_2_LETTER_ARPABET[syllable[i]])
        elif syllable[i] in t_phonemes :
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
  if i == 1 :
    break
  phn = phone_keys[i]
  # split TRAIN/TEST
  ## shuffle the words list
  random.shuffle(phoneme_to_words[phn]["words"])
  ## split the words list
  split_index = round(train_percentage * len(phoneme_to_words[phn]["words"]))
  split1 = set(phoneme_to_words[phn]["words"][:split_index])
  split2 = set(phoneme_to_words[phn]["words"][split_index:])
  ## assign words to train and test, ensuring no duplicates
  train = split1
  test = split2
  ## remove assigned words from other phoneme's word lists
  for other_phn, other_data in phoneme_to_words.items() :
    phoneme_to_words[other_phn]["words"] = list(set(other_data["words"]) - split2)

  # split TRAIN/VAL
  ## shuffle the words list
  random.shuffle(phoneme_to_words[phn]["words"])
  ## split the words list
  split_index = round(train_percentage * len(phoneme_to_words[phn]["words"]))
  split1 = set(phoneme_to_words[phn]["words"][:split_index])
  split2 = set(phoneme_to_words[phn]["words"][split_index:])
  ## assign words to train and val, ensuring no duplicates
  train = split1
  val = split2
  print(len(list(train)), train)
  print(len(list(val)), val)
  print(len(list(test)), test)
  ## remove assigned words from other phoneme's word lists
  for other_phn, other_data in phoneme_to_words.items() :
    phoneme_to_words[other_phn]["words"] = list(set(other_data["words"]) - split1 - split2)
  ## resort the phones based on their occurrences
  phoneme_to_words = dict(sorted(phoneme_to_words.items(), key=lambda item: len(item[1]["words"])))
  phone_keys = list(phoneme_to_words.keys())

print(phoneme_to_words.keys())
for phn, data in phoneme_to_words.items() :
  print(phn, len(data["words"]))

# # split TRAIN/TEST process end

# #TODO: SPLIT TRAIN/VAL MASIH SALAH
# phone_keys = list(phoneme_to_words.keys())
# # split TRAIN/VAL process start
# for i in range(len(phone_keys)) :
#   if i == 1 :
#     break
#   phn = phone_keys[i]
#   # print("before shuffle", phoneme_to_words[phn]["words"])
#   ## shuffle the words list
#   random.shuffle(phoneme_to_words[phn]["words"])
#   # print("after shuffle", phoneme_to_words[phn]["words"])
#   ## split the words list
#   split_index = round(train_percentage * len(phoneme_to_words[phn]["words"]))
#   # print(split_index)
#   split1 = set(phoneme_to_words[phn]["words"][:split_index])
#   split2 = set(phoneme_to_words[phn]["words"][split_index:])
#   # print(split1)
#   # print(split2)
#   ## assign words to train and test, ensuring no duplicates
#   train.update(split1 - val)
#   val.update(split2 - train)
#   # print(train)
#   # print(test)
#   ## remove assigned words from other phoneme's word lists
#   for other_phn, other_data in phoneme_to_words.items() :
#     phoneme_to_words[other_phn]["words"] = list(set(other_data["words"]) - split1 - split2)

#   ## resort the phones based on their occurrences
#   phoneme_to_words = dict(sorted(phoneme_to_words.items(), key=lambda item: len(item[1]["words"])))
#   phone_keys = list(phoneme_to_words.keys())

# split TRAIN/VAL process end


print(len(train), len(val), len(test))

# for key, value in dict(sorted(phoneme_to_words.items(), key=lambda item: item[1]["occurrences"])) :
#   print(key, value)

# with open(os.path.join(DATA_DIR, "ma/train.csv")) as f_read, open(os.path.join(DATA_DIR, "ma/train_converted.csv"), 'w') as f_write :
#   csv_reader = csv.reader(f_read)
#   csv_writer = csv.writer(f_write)
#   headers = next(csv_reader, None)
#   if headers :
#     csv_writer.writerow([headers[1]])
#   for row in csv_reader :
#     csv_writer.writerow([row[1]])