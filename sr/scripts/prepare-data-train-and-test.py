import os
import random
import re
import string
from typing import List, Tuple

random.seed(23522026)

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(CURR_DIR, "..", "dataset")
AUDIO_DIR = os.path.join(DATASET_DIR, "audio")
TRANSCRIPT_DIR = os.path.join(DATASET_DIR, "transcript")
DATA_DIR = os.path.join(CURR_DIR, "..", "data")
if not os.path.exists(DATA_DIR) :
  os.mkdir(DATA_DIR)
TRAIN_DIR = os.path.join(DATA_DIR, "train")
if not os.path.exists(TRAIN_DIR) :
  os.mkdir(TRAIN_DIR)
TEST_DIR = os.path.join(DATA_DIR, "test")
if not os.path.exists(TEST_DIR) :
  os.mkdir(TEST_DIR)

def split_data(dir_path, test_ratio=.2) -> Tuple[List, List] :
  # Get all file names in the `dir_path` directory
  all_files = [file for file in os.listdir(dir_path) if file.endswith(".wav")]
  all_files.sort()
  # Calculate the number of files for test split
  num_test_files = int(len(all_files) * test_ratio)
  # Randomly select files for test split
  test_files = random.sample(population=all_files, k=num_test_files)
  test_files.sort()
  # Create a list of training files
  train_files = [file for file in all_files if file not in test_files]
  train_files.sort()
  return train_files, test_files

def preprocess_text(text:str) -> str :
  """
    Returns cleaned text (no dashes, digits, tabs, punctuation, non-alphabetic characters)
  """
  # Remove <EN-EN>, </EN-EN>, and the sorts
  text = re.sub("<\/?\w+-\w+>", '', text)
  # Replace en em dashes with whitespace
  text = re.sub('–', ' ', text)
  text = re.sub('—', ' ', text)
  # Remove digits
  text = re.sub(r"\d", '', text)
  # Remove tabs
  text = re.sub(r"\t", '', text)
  # Remove non-alphabetic characters (except apostrophe and hyphen)
  text = re.sub(r"[^a-zA-Z\s\'-]", '', text)
  # Remove punctuation (except apostrophe and hyphen) using string.punctuation
  punctuation = f"{string.punctuation}‘’“”"
  punctuation = ''.join(char for char in punctuation if char not in "'-")
  text = text.translate(str.maketrans(punctuation, ' '*len(punctuation)))
  return text.lower().strip()

speaker_ids = sorted(os.listdir(AUDIO_DIR))

with open(os.path.join(TRAIN_DIR, "spk2gender"), 'w') as f_train_spk2gender,  \
     open(os.path.join(TRAIN_DIR, "wav.scp"), 'w') as f_train_wav_scp,        \
     open(os.path.join(TRAIN_DIR, "text"), 'w') as f_train_text,              \
     open(os.path.join(TRAIN_DIR, "utt2spk"), 'w') as f_train_utt2spk,        \
     open(os.path.join(TEST_DIR, "spk2gender"), 'w') as f_test_spk2gender,    \
     open(os.path.join(TEST_DIR, "wav.scp"), 'w') as f_test_wav_scp,          \
     open(os.path.join(TEST_DIR, "text"), 'w') as f_test_text,                \
     open(os.path.join(TEST_DIR, "utt2spk"), 'w') as f_test_utt2spk :
  for speaker_id in speaker_ids :
    train_files, test_files = split_data(os.path.join(AUDIO_DIR, speaker_id))
    if "female" in speaker_id : gender = 'f'
    elif "male" in speaker_id : gender = 'm'

    # Write `spk2gender` files (both train and test split)
    f_train_spk2gender.write(f"{speaker_id} {gender}\n")
    f_test_spk2gender.write(f"{speaker_id} {gender}\n")

    # Write `wav.scp`, `utt2spk`, and `text` files (train split)
    for file_name in train_files :
      # `wav.scp`
      utterance_id = f"{speaker_id}-{file_name.rstrip('.wav')}"
      f_train_wav_scp.write(f"{utterance_id} {os.path.join(AUDIO_DIR, speaker_id, file_name)}\n")
      # `utt2spk`
      f_train_utt2spk.write(f"{utterance_id} {speaker_id}\n")
      # `text`
      with open(os.path.join(TRANSCRIPT_DIR, speaker_id, f"{file_name.rstrip('.wav')}.txt")) as f_transcript :
        lines = f_transcript.readlines()
        if len(lines) > 1 : print(f"WARNING! {f'{utterance_id}.txt'} has more than one line of transcript")
        f_train_text.write(f"{utterance_id} {preprocess_text(lines[0]).rstrip().upper()}\n")

    # Write `wav.scp`, `utt2spk`, and `text` files (test split)
    for file_name in test_files :
      # `wav.scp`
      utterance_id = f"{speaker_id}-{file_name.rstrip('.wav')}"
      f_test_wav_scp.write(f"{utterance_id} {os.path.join(AUDIO_DIR, speaker_id, file_name)}\n")
      # `utt2spk`
      f_test_utt2spk.write(f"{utterance_id} {speaker_id}\n")
      # `text`
      with open(os.path.join(TRANSCRIPT_DIR, speaker_id, f"{file_name.rstrip('.wav')}.txt")) as f_transcript :
        lines = f_transcript.readlines()
        if len(lines) > 1 : print(f"WARNING! {f'{utterance_id}.txt'} has more than one line of transcript")
        f_test_text.write(f"{utterance_id} {preprocess_text(lines[0]).rstrip().upper()}\n")
