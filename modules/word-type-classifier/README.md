# `train.py`
To generate all trigrams and their probabilities in main language corpus file `data/train/main.txt` and foreign language corpus file `data/train/foreign.txt`.

Usage: `python3 train.py 3`

To train 3-gram

# `test.py`
To classify all texts inside the inputted test corpus file using n-grams from `ngrams/` trained by executing `train.py`.

Usage: `python3 test.py data/test/<filename> 2,3,4`

To classify all texts inside `data/test/<filename>` file using 2-gram, 3-gram, and 4-grams language model from `ngrams/` directory.
