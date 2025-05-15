import json
import os
import re
import shutil
import string

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(CURR_DIR, "..", "dataset")
AUDIO_DIR = os.path.join(DATASET_DIR, "audio")
if not os.path.exists(AUDIO_DIR) :
  os.mkdir(AUDIO_DIR)
TRANSCRIPT_DIR = os.path.join(DATASET_DIR, "transcript")
if not os.path.exists(TRANSCRIPT_DIR) :
  os.mkdir(TRANSCRIPT_DIR)
MEDISCO_DIR = os.path.join(DATASET_DIR, "MEDISCO")
MEDISCO_TRAIN_DIR = os.path.join(MEDISCO_DIR, "train")
MEDISCO_TEST_DIR = os.path.join(MEDISCO_DIR, "test")
DATA_DIR = os.path.join(CURR_DIR, "..", "data")
if not os.path.exists(DATA_DIR) :
  os.mkdir(DATA_DIR)
TRAIN_DIR = os.path.join(DATA_DIR, "train")
if not os.path.exists(TRAIN_DIR) :
  os.mkdir(TRAIN_DIR)
TEST_DIR = os.path.join(DATA_DIR, "test")
if not os.path.exists(TEST_DIR) :
  os.mkdir(TEST_DIR)

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

# Read all annotations from each of train and test directories
with open(os.path.join(MEDISCO_TRAIN_DIR, "annotation/annotated_train.json")) as json_train_annot, \
     open(os.path.join(MEDISCO_TEST_DIR, "annotation/annotated_test.json")) as json_test_annot :
  train_sent_id_to_transcription = json.load(json_train_annot)
  for k, v in train_sent_id_to_transcription.items() :
    train_sent_id_to_transcription[k] = preprocess_text(v)
  test_sent_id_to_transcription = json.load(json_test_annot)
  for k, v in test_sent_id_to_transcription.items() :
    test_sent_id_to_transcription[k] = preprocess_text(v)

# Read from MEDISCO's train subfolder
for train_speaker_id in os.listdir(os.path.join(MEDISCO_TRAIN_DIR, "speech")) :
  # Create `audio` and `transcript` folders for each speaker id
  if not os.path.exists(os.path.join(AUDIO_DIR, train_speaker_id)) :
    os.mkdir(os.path.join(AUDIO_DIR, train_speaker_id))
  if not os.path.exists(os.path.join(TRANSCRIPT_DIR, train_speaker_id)) :
    os.mkdir(os.path.join(TRANSCRIPT_DIR, train_speaker_id))
  # Copy *.wav audio files and create *.txt transcript files for each speaker id
  for wav_file in [file for file in os.listdir(os.path.join(MEDISCO_TRAIN_DIR, f"speech/{train_speaker_id}"))] :
    sent_id = wav_file.split('-')[1].rstrip(".wav")
    shutil.copy(
      os.path.join(MEDISCO_TRAIN_DIR, f"speech/{train_speaker_id}/{wav_file}"),
      os.path.join(AUDIO_DIR, f"{train_speaker_id}/{train_speaker_id}_{sent_id}.wav")
    )
    with open(os.path.join(TRANSCRIPT_DIR, f"{train_speaker_id}/{train_speaker_id}_{sent_id}.txt"), 'w') as f_transcript :
      f_transcript.write(train_sent_id_to_transcription[sent_id].strip())

# Read from MEDISCO's test subfolder
for test_speaker_id in os.listdir(os.path.join(MEDISCO_TEST_DIR, "speech")) :
  # Create `audio` and `transcript` folders for each speaker id
  if "female" in test_speaker_id :
    new_test_speaker_id = f"{test_speaker_id}-6"
  elif "male" in test_speaker_id :
    new_test_speaker_id = f"{test_speaker_id}-7"
  if not os.path.exists(os.path.join(AUDIO_DIR, new_test_speaker_id)) :
    os.mkdir(os.path.join(AUDIO_DIR, new_test_speaker_id))
  if not os.path.exists(os.path.join(TRANSCRIPT_DIR, new_test_speaker_id)) :
    os.mkdir(os.path.join(TRANSCRIPT_DIR, new_test_speaker_id))
  # Copy *.wav audio files and create *.txt transcript files for each speaker id
  for wav_file in [file for file in os.listdir(os.path.join(MEDISCO_TEST_DIR, f"speech/{test_speaker_id}"))] :
    sent_id = wav_file.split('-')[1].rstrip(".wav")
    shutil.copy(
      os.path.join(MEDISCO_TEST_DIR, f"speech/{test_speaker_id}/{wav_file}"),
      os.path.join(AUDIO_DIR, f"{new_test_speaker_id}/{new_test_speaker_id}_{sent_id}.wav")
    )
    with open(os.path.join(TRANSCRIPT_DIR, f"{new_test_speaker_id}/{new_test_speaker_id}_{sent_id}.txt"), 'w') as f_transcript :
      f_transcript.write(test_sent_id_to_transcription[sent_id].strip())
