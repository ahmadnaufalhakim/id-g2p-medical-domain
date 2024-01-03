import argparse
import csv
import datetime
import itertools
from kbbi import KBBI, TidakDitemukan, BatasSehari
import os
import re
from subprocess import call
import sys
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
CURRENT_TIMESTAMP = time.time()
CURRENT_TIMESTAMP_STR = datetime.datetime.fromtimestamp(CURRENT_TIMESTAMP).strftime("%Y-%m-%d_%H-%M-%S")
KBBI_API_LIMIT = False

parser = argparse.ArgumentParser()
parser.add_argument('entry_idx_start', help="entry index start (1-indexed)")
parser.add_argument('entry_idx_finish', help="entry index finish (1-indexed)")
args = parser.parse_args()

entry_idx_start = int(args.entry_idx_start)-1
entry_idx_finish = int(args.entry_idx_finish)-1
assert entry_idx_start <= entry_idx_finish

syllable_structures = [
  'V',      # a-kan
  "VC",     # in-dah
  "CV",     # bi-ru
  "VCC",    # eks
  "CVC",    # ram-but
  "CCV",    # pra-ba-yar
  "CVCC",   # teks
  "CCVC",   # prak-tik
  "CCCV",   # stra-te-gi
  "CVCCC",  # korps
  "CCVCC",  # kom-pleks
  "CCCVC"   # struk-tur
]
word_regex_patterns = {
  'C': "([b-df-hj-np-tv-z'])",
  'V': [
    "((ai)|(au)|(oi)|(ei))",
    "(a|i|u|e|o)"
  ]
}
phoneme_regex_patterns = {
  'C': [
    "((tʃ)|(dʒ)|(ŋ)|(ɲ)|(sj))",
    "((ʔ)|(b)|(d)|(f)|(g)|(h)|(k)|(l)|(m)|(n)|(p)|(r)|(s)|(t)|(v)|(w)|(j)|(z))"
  ],
  'V': [
    "((ai)|(au)|(oi)|(ei))",
    "(a|i|u|e|ə|o)"
  ]
}

def generate_word_syllable_sequence_candidates(word, syllables=[]) :
  if not word:
    # Base case: if the word is empty, we've found a valid combination of syllables
    return ['.'.join(syllables)]
  candidates = []
  for syllable_structure in syllable_structures :
    for V_regex_pattern in word_regex_patterns['V'] :
      pattern = re.compile(f"^{syllable_structure.replace('C', word_regex_patterns['C']).replace('V', V_regex_pattern)}")
      match = pattern.match(word)
      if match :
        # Extract the matched syllable
        syllable = match.group()
        remaining_word = word[len(syllable):]
        # Recursively generate candidates for the remaining part of the word
        candidates.extend(generate_word_syllable_sequence_candidates(remaining_word, syllables + [syllable]))
  return candidates

def generate_phoneme_syllable_sequence_candidates(phoneme, syllables=[]) :
  if not phoneme:
    # Base case: if the phoneme is empty, we've found a valid combination of syllables
    return ['.'.join(syllables)]
  candidates = []
  for syllable_structure in syllable_structures :
    C_positions = [i for i, char in enumerate(syllable_structure) if char == 'C']
    combinations = itertools.product(phoneme_regex_patterns['C'], repeat=len(C_positions))
    C_modified_syllable_structures = []
    for combination in combinations :
      syllable_structure_list = list(syllable_structure)
      for i, char in zip(C_positions, combination) :
        syllable_structure_list[i] = char
      C_modified_syllable_structures.append(''.join(syllable_structure_list))
    for C_modified_syllable_structure in C_modified_syllable_structures :
      for V_regex_pattern in phoneme_regex_patterns['V'] :
        pattern = re.compile(f"^{C_modified_syllable_structure.replace('V', V_regex_pattern)}")
        match = pattern.match(phoneme)
        if match :
          # Extract the matched syllable
          syllable = match.group()
          remaining_phoneme = phoneme[len(syllable):]
          # Recursively generate candidates for the remaining part of the phoneme
          candidates.extend(generate_phoneme_syllable_sequence_candidates(remaining_phoneme, syllables + [syllable]))
  return candidates

def clear() :
  _ = call("clear" if os.name == "posix" else "cls")

with open(os.path.join(DATA_DIR, "ma/train_1_1.csv")) as f_read,\
     open(os.path.join(DATA_DIR, f"ma/train_{args.entry_idx_start}-{args.entry_idx_finish}_{CURRENT_TIMESTAMP_STR}.csv"), 'w') as f_write :
  csv_reader = csv.reader(f_read)
  csv_writer = csv.writer(f_write)
  rows = list(csv_reader)

  for i in range(entry_idx_start, entry_idx_finish+1) :
    print(f"\n{i+1} out of {len(rows)} data ({round(100*(i+1)/len(rows),3)}%)") 
    print(rows[i][0])
    kbbi_word_entry_choice = -1
    word_entry_choice = -1
    phoneme_entry_choice = -1
    word_entry = ''
    phoneme_entry = ''
    if not KBBI_API_LIMIT :
      try :
        kbbi_word_entry_candidates = KBBI(rows[i][0]).entri
        print("#######################")
        print("## KBBI word entries ##")
        print("#######################")
        for j, kbbi_word_entry_candidate in enumerate(kbbi_word_entry_candidates) :
          print(j, kbbi_word_entry_candidate.nama, kbbi_word_entry_candidate.pelafalan)
        print("Choose which KBBI entry to be written as the word syllable sequence")
        kbbi_word_entry_choice = input(f"Enter the entry number 0{f'-{len(kbbi_word_entry_candidates)-1}' if len(kbbi_word_entry_candidates) > 1 else ''} (or 's' to skip all KBBI entries): ")
        if kbbi_word_entry_choice.isnumeric() :
          kbbi_word_entry_choice = int(kbbi_word_entry_choice)
        while kbbi_word_entry_choice not in range(0,len(kbbi_word_entry_candidates)) and kbbi_word_entry_choice != 's' :
          kbbi_word_entry_choice = input(f"Entry invalid. Enter the entry number 0{f'-{len(kbbi_word_entry_candidates)-1}' if len(kbbi_word_entry_candidates) > 1 else ''} (or 's' to skip all KBBI entries): ")
          if kbbi_word_entry_choice.isnumeric() :
            kbbi_word_entry_choice = int(kbbi_word_entry_choice)
        if kbbi_word_entry_choice != 's' :
          word_entry = kbbi_word_entry_candidates[kbbi_word_entry_choice].nama
        else :
          word_entry_candidates = generate_word_syllable_sequence_candidates(rows[i][0])
          if len(word_entry_candidates) > 0 :
            print("############################")
            print("## Generated word entries ##")
            print("############################")
            for j, word_entry_candidate in enumerate(word_entry_candidates) :
              print(j, word_entry_candidate)
            print("Choose which gen'd entry to be written as the word syllable sequence")
            word_entry_choice = input(f"Enter the entry number 0{f'-{len(word_entry_candidates)-1}' if len(word_entry_candidates) > 1 else ''} (or 's' to skip all gen'd entries): ")
            if word_entry_choice.isnumeric() :
              word_entry_choice = int(word_entry_choice)
            while word_entry_choice not in range(0,len(word_entry_candidates)) and word_entry_choice != 's' :
              word_entry_choice = input(f"Entry invalid. Enter the entry number 0{f'-{len(word_entry_candidates)-1}' if len(word_entry_candidates) > 1 else ''} (or 's' to skip all gen'd entries): ")
              if word_entry_choice.isnumeric() :
                word_entry_choice = int(word_entry_choice)
            if word_entry_choice != 's' :
              word_entry = word_entry_candidates[word_entry_choice]
            else :
              word_entry = input(f"Enter your custom word entry syllable sequence: ")
          else :
            print("Cannot generate word syllable sequence candidates")
            word_entry = input(f"Enter your custom word entry syllable sequence: ")
      except (TidakDitemukan, BatasSehari) as e :
        print(f"An exception occurred: {e}")
        if type(e).__name__ == "BatasSehari" :
          KBBI_API_LIMIT = True
        word_entry_candidates = generate_word_syllable_sequence_candidates(rows[i][0])
        if len(word_entry_candidates) > 0 :
          print("############################")
          print("## Generated word entries ##")
          print("############################")
          for j, word_entry_candidate in enumerate(word_entry_candidates) :
            print(j, word_entry_candidate)
          print("Choose which gen'd entry to be written as the word syllable sequence")
          word_entry_choice = input(f"Enter the entry number 0{f'-{len(word_entry_candidates)-1}' if len(word_entry_candidates) > 1 else ''} (or 's' to skip all gen'd entries): ")
          if word_entry_choice.isnumeric() :
            word_entry_choice = int(word_entry_choice)
          while word_entry_choice not in range(0,len(word_entry_candidates)) and word_entry_choice != 's' :
            word_entry_choice = input(f"Entry invalid. Enter the entry number 0{f'-{len(word_entry_candidates)-1}' if len(word_entry_candidates) > 1 else ''} (or 's' to skip all gen'd entries): ")
            if word_entry_choice.isnumeric() :
              word_entry_choice = int(word_entry_choice)
          if word_entry_choice != 's' :
            word_entry = word_entry_candidates[word_entry_choice]
          else :
            word_entry = input(f"Enter your custom word entry syllable sequence: ")
        else :
          print("Cannot generate word syllable sequence candidates")
          word_entry = input(f"Enter your custom word entry syllable sequence: ")
    else :
      word_entry_candidates = generate_word_syllable_sequence_candidates(rows[i][0])
      if len(word_entry_candidates) > 0 :
        print("############################")
        print("## Generated word entries ##")
        print("############################")
        for j, word_entry_candidate in enumerate(word_entry_candidates) :
          print(j, word_entry_candidate)
        print("Choose which gen'd entry to be written as the word syllable sequence")
        word_entry_choice = input(f"Enter the entry number 0-{len(word_entry_candidates)-1} (or 's' to skip all gen'd entries): ")
        if word_entry_choice.isnumeric() :
          word_entry_choice = int(word_entry_choice)
        while word_entry_choice not in range(0,len(word_entry_candidates)) and word_entry_choice != 's' :
          word_entry_choice = input(f"Entry invalid. Enter the entry number 0-{len(word_entry_candidates)-1} (or 's' to skip all gen'd entries): ")
          if word_entry_choice.isnumeric() :
            word_entry_choice = int(word_entry_choice)
        if word_entry_choice != 's' :
          word_entry = word_entry_candidates[word_entry_choice]
        else :
          word_entry = input(f"Enter your custom word entry syllable sequence: ")
      else :
        print("Cannot generate word syllable sequence candidates")
        word_entry = input(f"Enter your custom word entry syllable sequence: ")

    phoneme_entry_candidates = generate_phoneme_syllable_sequence_candidates(rows[i][1])
    if len(phoneme_entry_candidates) > 0 :
      print("###############################")
      print("## Generated phoneme entries ##")
      print("###############################")
      for j, phoneme_entry_candidate in enumerate(phoneme_entry_candidates) :
        print(j, phoneme_entry_candidate)
      print("Choose which gen'd entry to be written as the phoneme syllable sequence")
      phoneme_entry_choice = input(f"Enter the entry number 0{f'-{len(phoneme_entry_candidates)-1}' if len(phoneme_entry_candidates) > 1 else ''} (or 's' to skip all gen'd entries): ")
      if phoneme_entry_choice.isnumeric() :
        phoneme_entry_choice = int(phoneme_entry_choice)
      while phoneme_entry_choice not in range(0,len(phoneme_entry_candidates)) and phoneme_entry_choice != 's' :
        phoneme_entry_choice = input(f"Entry invalid. Enter the entry number 0{f'-{len(phoneme_entry_candidates)-1}' if len(phoneme_entry_candidates) > 1 else ''} (or 's' to skip all gen'd entries): ")
        if phoneme_entry_choice.isnumeric() :
          phoneme_entry_choice = int(phoneme_entry_choice)
      if phoneme_entry_choice != 's' :
        phoneme_entry = phoneme_entry_candidates[phoneme_entry_choice]
      else :
        phoneme_entry = input(f"Enter your custom phoneme entry syllable sequence: ")
    else :
      print("Cannot generate phoneme syllable sequence candidates")
      phoneme_entry = input(f"Enter your custom phoneme entry syllable sequence: ")

    csv_writer.writerow([i+1] + rows[i] + [word_entry, phoneme_entry])
    kbbi_word_entry_choice = -1
    word_entry_choice = -1
    word_entry = ''

    sys.stdout.flush()
    # time.sleep(1)
    # clear()
