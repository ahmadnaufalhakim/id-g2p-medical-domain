import csv
import os
import random

random.seed(23522026)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

ma_train_words = set(); train_rows = set()
ma_val_words = set(); val_rows = set()
ma_test_words = set(); test_rows = set()
with open(os.path.join(DATA_DIR, "en_ma/en_train_converted.csv")) as en_train_csv_read, \
     open(os.path.join(DATA_DIR, "en_ma/ma_train_converted.csv")) as ma_train_csv_read, \
     open(os.path.join(DATA_DIR, "en_ma/en_val_converted.csv")) as en_val_csv_read, \
     open(os.path.join(DATA_DIR, "en_ma/ma_val_converted.csv")) as ma_val_csv_read, \
     open(os.path.join(DATA_DIR, "en_ma/en_test_converted.csv")) as en_test_csv_read, \
     open(os.path.join(DATA_DIR, "en_ma/ma_test_converted.csv")) as ma_test_csv_read, \
     open(os.path.join(DATA_DIR, "en_ma/train_converted.csv"), 'w') as f_train_write, \
     open(os.path.join(DATA_DIR, "en_ma/val_converted.csv"), 'w') as f_val_write, \
     open(os.path.join(DATA_DIR, "en_ma/test_converted.csv"), 'w') as f_test_write :
  en_train_csv_reader = csv.reader(en_train_csv_read)
  ma_train_csv_reader = csv.reader(ma_train_csv_read)
  en_val_csv_reader = csv.reader(en_val_csv_read)
  ma_val_csv_reader = csv.reader(ma_val_csv_read)
  en_test_csv_reader = csv.reader(en_test_csv_read)
  ma_test_csv_reader = csv.reader(ma_test_csv_read)
  train_csv_writer = csv.writer(f_train_write)
  val_csv_writer = csv.writer(f_val_write)
  test_csv_writer = csv.writer(f_test_write)

  # Skip headers
  next(en_train_csv_reader, None)
  next(ma_train_csv_reader, None)
  next(en_val_csv_reader, None)
  next(ma_val_csv_reader, None)
  next(en_test_csv_reader, None)
  next(ma_test_csv_reader, None)

  for row in ma_train_csv_reader :
    train_rows.add((row[0].lower(), row[-1], "ma"))
    ma_train_words.add(row[0])
  for row in ma_val_csv_reader :
    val_rows.add((row[0].lower(), row[-1], "ma"))
    ma_val_words.add(row[0])
  for row in ma_test_csv_reader :
    test_rows.add((row[0].lower(), row[-1], "ma"))  
    ma_test_words.add(row[0])
  for row in en_train_csv_reader :
    if row[0].lower() not in ma_train_words and \
       row[0].lower() not in ma_val_words and \
       row[0].lower() not in ma_test_words :
      train_rows.add((row[0].lower(), row[-1], "en"))
  for row in en_val_csv_reader :
    if row[0].lower() not in ma_train_words and \
       row[0].lower() not in ma_val_words and \
       row[0].lower() not in ma_test_words :
      val_rows.add((row[0].lower(), row[-1], "en"))
  for row in en_test_csv_reader :
    if row[0].lower() not in ma_train_words and \
       row[0].lower() not in ma_val_words and \
       row[0].lower() not in ma_test_words :
      test_rows.add((row[0].lower(), row[-1], "en"))

  train_csv_writer.writerow(["word", "arpabet_phoneme_sequence", "lang"])
  val_csv_writer.writerow(["word", "arpabet_phoneme_sequence", "lang"])
  test_csv_writer.writerow(["word", "arpabet_phoneme_sequence", "lang"])

  for train_row in sorted(list(train_rows), key=lambda row: row[0]) :
    train_csv_writer.writerow([train_row[0], train_row[1], train_row[2]])
  for val_row in sorted(list(val_rows), key=lambda row: row[0]) :
    val_csv_writer.writerow([val_row[0], val_row[1], val_row[2]])
  for test_row in sorted(list(test_rows), key=lambda row: row[0]) :
    test_csv_writer.writerow([test_row[0], test_row[1], test_row[2]])
