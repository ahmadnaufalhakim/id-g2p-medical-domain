"""
Prepare data for word type classifier module training.
Essentially copies all the words data used from `modules/prn_gen/data` (en_ma).
"""
import csv
import os

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SOURCE_DIR = os.path.join(CURR_DIR, "../../prn_gen/data")
DATA_DIR = os.path.join(CURR_DIR, "..", "data")
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")
if not os.path.exists(TRAIN_DATA_DIR) :
  os.mkdir(TRAIN_DATA_DIR)
VAL_DATA_DIR = os.path.join(DATA_DIR, "val")
if not os.path.exists(VAL_DATA_DIR) :
  os.mkdir(VAL_DATA_DIR)
TEST_DATA_DIR = os.path.join(DATA_DIR, "test")
if not os.path.exists(TEST_DATA_DIR) :
  os.mkdir(TEST_DATA_DIR)

if __name__ == "__main__" :
  """CORE OF THE SCRIPT"""
  train_foreign_words = set(); train_main_words = set()
  val_foreign_words = set(); val_main_words = set()
  test_foreign_words = set(); test_main_words = set()
  with open(os.path.join(DATA_SOURCE_DIR, "en_ma/en_train_converted.csv")) as en_train_csv_read, \
       open(os.path.join(DATA_SOURCE_DIR, "en_ma/ma_train_converted.csv")) as ma_train_csv_read, \
       open(os.path.join(DATA_SOURCE_DIR, "en_ma/en_val_converted.csv")) as en_val_csv_read, \
       open(os.path.join(DATA_SOURCE_DIR, "en_ma/ma_val_converted.csv")) as ma_val_csv_read, \
       open(os.path.join(DATA_SOURCE_DIR, "en_ma/en_test_converted.csv")) as en_test_csv_read, \
       open(os.path.join(DATA_SOURCE_DIR, "en_ma/ma_test_converted.csv")) as ma_test_csv_read, \
       open(os.path.join(TRAIN_DATA_DIR, "foreign.csv"), 'w') as train_foreign_write, \
       open(os.path.join(TRAIN_DATA_DIR, "main.csv"), 'w') as train_main_write, \
       open(os.path.join(VAL_DATA_DIR, "foreign.csv"), 'w') as val_foreign_write, \
       open(os.path.join(VAL_DATA_DIR, "main.csv"), 'w') as val_main_write, \
       open(os.path.join(TEST_DATA_DIR, "foreign.csv"), 'w') as test_foreign_write, \
       open(os.path.join(TEST_DATA_DIR, "main.csv"), 'w') as test_main_write :
    en_train_csv_reader = csv.reader(en_train_csv_read)
    ma_train_csv_reader = csv.reader(ma_train_csv_read)
    en_val_csv_reader = csv.reader(en_val_csv_read)
    ma_val_csv_reader = csv.reader(ma_val_csv_read)
    en_test_csv_reader = csv.reader(en_test_csv_read)
    ma_test_csv_reader = csv.reader(ma_test_csv_read)
    train_foreign_csv_writer = csv.writer(train_foreign_write)
    train_main_csv_writer = csv.writer(train_main_write)
    val_foreign_csv_writer = csv.writer(val_foreign_write)
    val_main_csv_writer = csv.writer(val_main_write)
    test_foreign_csv_writer = csv.writer(test_foreign_write)
    test_main_csv_writer = csv.writer(test_main_write)

    # Skip headers
    next(en_train_csv_reader, None)
    next(ma_train_csv_reader, None)
    next(en_val_csv_reader, None)
    next(ma_val_csv_reader, None)
    next(en_test_csv_reader, None)
    next(ma_test_csv_reader, None)

    # Store all unique words from `modules/prn_gen/data`
    for row in en_train_csv_reader :
      train_foreign_words.add(row[0].lower())
    for row in ma_train_csv_reader :
      train_main_words.add(row[0].lower())
    for row in en_val_csv_reader :
      val_foreign_words.add(row[0].lower())
    for row in ma_val_csv_reader :
      val_main_words.add(row[0].lower())
    for row in en_test_csv_reader :
      test_foreign_words.add(row[0].lower())
    for row in ma_test_csv_reader :
      test_main_words.add(row[0].lower())

    # Write all unique words to the data in `word_type_classifier` module directory
    for train_foreign_row in sorted(list(train_foreign_words)) :
      train_foreign_csv_writer.writerow([train_foreign_row])
    for train_main_row in sorted(list(train_main_words)) :
      train_main_csv_writer.writerow([train_main_row])
    for val_foreign_row in sorted(list(val_foreign_words)) :
      val_foreign_csv_writer.writerow([val_foreign_row])
    for val_main_row in sorted(list(val_main_words)) :
      val_main_csv_writer.writerow([val_main_row])
    for test_foreign_row in sorted(list(test_foreign_words)) :
      test_foreign_csv_writer.writerow([test_foreign_row])
    for test_main_row in sorted(list(test_main_words)) :
      test_main_csv_writer.writerow([test_main_row])
