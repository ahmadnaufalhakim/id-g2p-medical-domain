import csv
import os
import random
import re
import sys
import traceback

random.seed(23522026)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

DEFAULT_ARPABET_TO_IPA = {
  "AA": 'a',
  "AE": 'e',
  "AH": 'ə',
  "AO": 'o',
  "AW": "au",
  "AY": "ai",
  'B': 'b',
  "CH": "tʃ",
  'D': 'd',
  "DH": 'd',
  "EH": 'e',
  "ER": "ə r",
  "EY": "ei",
  'F': 'f',
  'G': 'g',
  "HH": 'h',
  "IH": 'i',
  "IY": 'i',
  "JH": "dʒ",
  'K': 'k',
  'L': 'l',
  'M': 'm',
  'N': 'n',
  "NG": 'ŋ',
  "OW": "o w",
  "OY": "oi",
  'P': 'p',
  'R': 'r',
  'S': 's',
  "SH": "sj",
  'T': 't',
  "TH": 't',
  "UH": 'u',
  "UW": "u w",
  'V': 'v',
  'W': 'w',
  'Y': 'j',
  'Z': 'z',
  "ZH": 'z',
}

# all vocal arpabets
VOCAL_ARPABETS = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "EY", "IH", "IY", "OY", "UH"]
AH_IH_ARPABETS = ["AH", "IH"]
# vocal arpabets minus AH and IH
AH_IH_EXCLUDED_VOCAL_ARPABETS = [arpabet for arpabet in VOCAL_ARPABETS if arpabet not in AH_IH_ARPABETS]
# vocal arpabets + ER OW UW
VOCAL_STARTING_ARPABETS = VOCAL_ARPABETS[:]
VOCAL_STARTING_ARPABETS.extend(["ER", "OW", "UW"])

# all consonant arpabets
CONSONANT_ARPABETS = ['B', "CH", 'D', "DH", 'F', 'G', "HH", "JH", 'K', 'L', 'M', 'N', "NG", 'P', 'R', 'S', "SH", 'T', "TH", 'V', 'W', 'Y', 'Z', "ZH"]
HH_L_R_W_Y_ARPABETS = ["HH", 'L', 'R', 'W', 'Y']
# consonant arpabets minus HH, L, R, W, and Y
HH_L_R_W_Y_EXCLUDED_CONSONANT_ARPABETS = [arpabet for arpabet in CONSONANT_ARPABETS if arpabet not in HH_L_R_W_Y_ARPABETS]
# consonant arpabets + ER OW UW
CONSONANT_ENDING_ARPABETS = CONSONANT_ARPABETS[:]
CONSONANT_ENDING_ARPABETS.extend(["ER", "OW", "UW"])

# 'R' ending arpabets
R_ENDING_ARPABETS = ["ER", 'R']

IPA_TO_2_LETTER_ARPABET = {
  'ʔ': 'Q',
  'a': "AA",
  "ai": "AY", # diftong ai
  "au": "AW", # diftong au
  'b': 'B',
  "tʃ": "CH", # c
  'd': 'D',
  'e': "EH",
  "ei": "EY", # diftong ei
  'ə': "AX",
  'f': 'F',
  'g': 'G',
  'h': "HH",
  'i': "IY",
  "dʒ": "JH", # j
  'k': 'K',
  'l': 'L',
  'm': 'M',
  'n': 'N',
  "oi": "OY", # diftong oi
  'o': "AO",
  'ŋ': "NG",  # ng
  'ɲ': "NY",  # ny
  'p': 'P',
  'r': 'R',
  's': 'S',
  "sj": "SH", # sh,sy
  't': 'T',
  'u': "UW",
  'v': 'V',
  'w': 'W',
  'j': 'Y',  # y
  'z': 'Z',
}

rows = set()

with open(os.path.join(DATA_DIR, "en/train.csv")) as train_csv_read, \
     open(os.path.join(DATA_DIR, "en/validation.csv")) as val_csv_read, \
     open(os.path.join(DATA_DIR, "en/test.csv")) as test_csv_read :
  train_csv_reader = csv.reader(train_csv_read)
  val_csv_reader = csv.reader(val_csv_read)
  test_csv_reader = csv.reader(test_csv_read)

  # Skip headers
  next(train_csv_reader, None)
  next(val_csv_reader, None)
  next(test_csv_reader, None)

  # Add rows from train val test
  rows.update(tuple(row) for row in train_csv_reader)
  rows.update(tuple(row) for row in val_csv_reader)
  rows.update(tuple(row) for row in test_csv_reader)

  rows = sorted(rows)

  for row in rows :
    grapheme = row[0]
    arpabet_phoneme_sequence = row[1].split()
    ipa_phoneme_sequence = []
    ONE_PHN_COND = lambda i, rule_found : i <= len(arpabet_phoneme_sequence)-1 and not rule_found
    TWO_PHN_COND = lambda i, rule_found : i+1 <= len(arpabet_phoneme_sequence)-1 and not rule_found
    THREE_PHN_COND = lambda i, rule_found : i+2 <= len(arpabet_phoneme_sequence)-1 and not rule_found
    obs_flag = None # to observe certain phoneme rules
                    # use `obs_flag = True` in to be observed phoneme rule
    i = 0
    while i<len(arpabet_phoneme_sequence) :
      rule_found_flag = False
      try :
        # <sos>D AE => d ə (<sos>D'A in grapheme)
        # <sos>D IH => d i (<sos>DI in grapheme)
        # <sos>D IH => d ə (<sos>DE in grapheme)
        if TWO_PHN_COND(i, rule_found_flag) and \
           arpabet_phoneme_sequence[i] == 'D' and i==0 :
          if arpabet_phoneme_sequence[i+1] == "AE" and \
             grapheme.startswith("D'A") :
            # obs_flag = True
            ipa_phoneme_sequence.extend(['d', 'ə'])
            i += 2; rule_found_flag = True
          elif arpabet_phoneme_sequence[i+1] == "IH" :
            if grapheme.startswith("DI") :
              # obs_flag = True
              ipa_phoneme_sequence.extend(['d', 'i'])
              i += 2; rule_found_flag = True
            elif grapheme.startswith("DE") :
              # obs_flag = True
              ipa_phoneme_sequence.extend(['d', 'ə'])
              i += 2; rule_found_flag = True
        # <consonant-ending> G|K <consonant> => <corresp-consonant-ending> K <corresp-consonant>
        # WINGLER, TRANSGRESSOR, WORKWEEK, WITCHCRAFT, etc.
        if THREE_PHN_COND(i, rule_found_flag) and \
           arpabet_phoneme_sequence[i] in CONSONANT_ENDING_ARPABETS and \
           arpabet_phoneme_sequence[i+1] in ['G', 'K'] and \
           arpabet_phoneme_sequence[i+2] in CONSONANT_ARPABETS :
          # obs_flag = True
          ipa_phoneme_sequence.extend([
            DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i]],
            DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i+1]],
            DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i+2]]
          ])
          i += 3; rule_found_flag = True
        # <vocal-except-AH-IH> G|K <consonant-except-HH-L-R-W-Y> => <corresp-vocal-except-AH-IH> ʔ <corresp-consonant-except-HH-L-R-W-Y>
        # <vocal-except-AH-IH> G|K <HH-L-R-W-Y> => <corresp-vocal-except-AH-IH> g|k <corresp-HH-L-R-W-Y>
        #TODO: need checking if VOCAL_ARPABETS should be changed to AH_IH_EXCLUDED_VOCAL_ARPABETS
        if THREE_PHN_COND(i, rule_found_flag) and \
           arpabet_phoneme_sequence[i] in AH_IH_EXCLUDED_VOCAL_ARPABETS :
          if arpabet_phoneme_sequence[i+1] in ['G', 'K'] :
            if arpabet_phoneme_sequence[i+2] in HH_L_R_W_Y_EXCLUDED_CONSONANT_ARPABETS :
              # obs_flag = True
              ipa_phoneme_sequence.extend([
                DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i]],
                'ʔ',
                DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i+2]]
              ])
              i += 3; rule_found_flag = True
            elif arpabet_phoneme_sequence[i+2] in HH_L_R_W_Y_ARPABETS :
              # obs_flag = True
              ipa_phoneme_sequence.extend([
                DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i]],
                DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i+1]],
                DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i+2]]
              ])
              i += 3; rule_found_flag = True
        elif THREE_PHN_COND(i, rule_found_flag) and \
             arpabet_phoneme_sequence[i] in AH_IH_ARPABETS :
          if arpabet_phoneme_sequence[i] == "AH" :
            if arpabet_phoneme_sequence[i+1] == 'G' :
              if arpabet_phoneme_sequence[i+2] in VOCAL_STARTING_ARPABETS :
                # obs_flag = True
                ag_patterns = [
                  re.compile(r"^([JT])?UG"),
                  re.compile(r"([BDFGHKMN])UG"),
                  re.compile(r"(?<!R)RUG"),
                  re.compile(r"(?<![AEIOU])([B-DF-HJ-NP-TV-Z])?LUG")
                ]
                n_match = sum(bool(ag_pattern.search(grapheme)) for ag_pattern in ag_patterns)
                if n_match < 1 :
                  ipa_phoneme_sequence.extend(['ə', 'g'])
                  i += 2; rule_found_flag = True
                else :
                  ipa_phoneme_sequence.extend(['a', 'g'])
                  i += 2; rule_found_flag = True
              elif arpabet_phoneme_sequence[i+2] in HH_L_R_W_Y_ARPABETS :
                # obs_flag = True
                ag_patterns = [
                  re.compile(r"^UG"),
                  re.compile(r"([HJLMNOPST])UG"),
                  re.compile(r"(?<![AEIOU])RUG")
                ]
                n_match = sum(bool(ag_pattern.search(grapheme)) for ag_pattern in ag_patterns)
                if n_match < 1 :
                  ipa_phoneme_sequence.extend([
                    'ə', 'g',
                    DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i+2]]
                  ])
                  i += 3; rule_found_flag = True
                else :
                  ipa_phoneme_sequence.extend([
                    'a', 'g',
                    DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i+2]]
                  ])
                  i += 3; rule_found_flag = True
              elif arpabet_phoneme_sequence[i+2] in HH_L_R_W_Y_EXCLUDED_CONSONANT_ARPABETS :
                # obs_flag = True
                aʔ_pattern = re.compile(r"([BDHJLMNOPRSTZ])UG")
                if aʔ_pattern.search(grapheme) :
                  ipa_phoneme_sequence.extend([
                    'a', 'ʔ',
                    DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i+2]]
                  ])
                  i += 3; rule_found_flag = True
                else :
                  ipa_phoneme_sequence.extend([
                    'ə', 'ʔ',
                    DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i+2]]
                  ])
                  i += 3; rule_found_flag = True
            elif arpabet_phoneme_sequence[i+1] == 'K' :
              if arpabet_phoneme_sequence[i+2] in VOCAL_STARTING_ARPABETS :
                # obs_flag = True
                ak_patterns = [
                  re.compile(r"([BLY])UCC"),
                  re.compile(r"([BDHKRT])UCH"),
                  re.compile(r"SUCHAN"),
                  re.compile(r"^(?!HORN).*UCK"),
                  re.compile(r"LUKAC(H|S)"),
                  re.compile(r"TUK")
                ]
                n_match = sum(bool(ak_pattern.search(grapheme)) for ak_pattern in ak_patterns)
                if n_match < 1 :
                  ipa_phoneme_sequence.extend(['ə', 'k'])
                  i += 2; rule_found_flag = True
                else :
                  ipa_phoneme_sequence.extend(['a', 'k'])
                  i += 2; rule_found_flag = True
              elif arpabet_phoneme_sequence[i+2] in HH_L_R_W_Y_ARPABETS :
                # obs_flag = True
                ak_patterns = [
                  re.compile(r"UCC"),
                  re.compile(r"^(?!MC).*UCH"),
                  re.compile(r"^(?!MC).*UCK"),
                  re.compile(r"([BM])UK"),
                  re.compile(r"KUKLA")
                ]
                n_match = sum(bool(ak_pattern.search(grapheme)) for ak_pattern in ak_patterns)
                if n_match < 1 :
                  ipa_phoneme_sequence.extend([
                    'ə', 'k',
                    DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i+2]]
                  ])
                  i += 3; rule_found_flag = True
                else :
                  ipa_phoneme_sequence.extend([
                    'a', 'k',
                    DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i+2]]
                  ])
                  i += 3; rule_found_flag = True
              elif arpabet_phoneme_sequence[i+2] in HH_L_R_W_Y_EXCLUDED_CONSONANT_ARPABETS :
                # obs_flag = True
                aʔ_patterns = [
                  re.compile(r"UCH"),
                  re.compile(r"UCK"),
                  re.compile(r"UCT"),
                  re.compile(r"UK")
                ]
                n_match = sum(bool(aʔ_pattern.search(grapheme)) for aʔ_pattern in aʔ_patterns)
                if n_match < 1 :
                  ipa_phoneme_sequence.extend([
                    'ə', 'ʔ',
                    DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i+2]]
                  ])
                  i += 3; rule_found_flag = True
                else :
                  ipa_phoneme_sequence.extend([
                    'a', 'ʔ',
                    DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i+2]]
                  ])
                  i += 3; rule_found_flag = True
          elif arpabet_phoneme_sequence[i] == "IH" :
            if arpabet_phoneme_sequence[i+1] == 'G' :
              if arpabet_phoneme_sequence[i+2] in VOCAL_STARTING_ARPABETS :
                # obs_flag = True
                əg_patterns = [
                  re.compile(r"AG"),
                  re.compile(r"(?!^)EG")
                ]
                n_match = sum(bool(əg_pattern.search(grapheme)) for əg_pattern in əg_patterns)
                if n_match < 1 :
                  ipa_phoneme_sequence.extend(['i', 'g'])
                  i += 2; rule_found_flag = True
                else :
                  ipa_phoneme_sequence.extend(['ə', 'g'])
                  i += 2; rule_found_flag = True
              elif arpabet_phoneme_sequence[i+2] in HH_L_R_W_Y_ARPABETS :
                # obs_flag = True
                əg_pattern = re.compile(r"(?!^EGRESS)EG")
                if əg_pattern.search(grapheme) :
                  ipa_phoneme_sequence.extend([
                    'ə', 'g',
                    DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i+2]]
                  ])
                  i += 3; rule_found_flag = True
                else :
                  ipa_phoneme_sequence.extend([
                    'i', 'g',
                    DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i+2]]
                  ])
                  i += 3; rule_found_flag = True
              elif arpabet_phoneme_sequence[i+2] in HH_L_R_W_Y_EXCLUDED_CONSONANT_ARPABETS :
                # obs_flag = True
                if i==0 :
                  əʔ_pattern = re.compile(r"^E?X")
                  if əʔ_pattern.search(grapheme) :
                    ipa_phoneme_sequence.extend([
                      'ə', 'ʔ',
                      DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i+2]]
                    ])
                    i += 3; rule_found_flag = True
                  else :
                    ipa_phoneme_sequence.extend([
                      'i', 'ʔ',
                      DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i+2]]
                    ])
                    i += 3; rule_found_flag = True
                else :
                  əʔ_patterns = [
                    re.compile(r"(?<![(GN)|(EX)|Z)])AG"),
                    re.compile(r"OG")
                  ]
                  n_match = sum(bool(əʔ_pattern.search(grapheme)) for əʔ_pattern in əʔ_patterns)
                  if n_match < 1 :
                    ipa_phoneme_sequence.extend([
                      'i', 'ʔ',
                      DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i+2]]
                    ])
                    i += 3; rule_found_flag = True
                  else :
                    ipa_phoneme_sequence.extend([
                      'ə', 'ʔ',
                      DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i+2]]
                    ])
                    i += 3; rule_found_flag = True
            elif arpabet_phoneme_sequence[i+1] == 'K' :
              if arpabet_phoneme_sequence[i+2] in VOCAL_STARTING_ARPABETS :
                # obs_flag = True
                ək_patterns = [
                  re.compile(r"EC[AEIOU]"),
                  re.compile(r"IC(?!C|K)")
                ]
                n_match = sum(bool(ək_pattern.search(grapheme)) for ək_pattern in ək_patterns)
                if n_match < 1 :
                  ipa_phoneme_sequence.extend(['i', 'k'])
                  i += 2; rule_found_flag = True
                else :
                  ipa_phoneme_sequence.extend(['ə', 'k'])
                  i += 2; rule_found_flag = True
              elif arpabet_phoneme_sequence[i+2] in HH_L_R_W_Y_ARPABETS :
                # obs_flag = True
                ək_patterns = [
                  re.compile(r"EC[AEIOUR]"),
                  re.compile(r"EQ"),
                  re.compile(r"IC(?!H|K)(AL|L|U)")
                ]
                n_match = sum(bool(ək_pattern.search(grapheme)) for ək_pattern in ək_patterns)
                if n_match < 1 :
                  ipa_phoneme_sequence.extend([
                    'i', 'k',
                    DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i+2]]
                  ])
                  i += 3; rule_found_flag = True
                else :
                  ipa_phoneme_sequence.extend([
                    'ə', 'k',
                    DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i+2]]
                  ])
                  i += 3; rule_found_flag = True
              elif arpabet_phoneme_sequence[i+2] in HH_L_R_W_Y_EXCLUDED_CONSONANT_ARPABETS :
                # obs_flag = True
                əʔ_patterns = [
                  re.compile(r"EC"),
                  re.compile(r"^(?!.*ICS).*EX.*")
                ]
                n_match = sum(bool(əʔ_pattern.search(grapheme)) for əʔ_pattern in əʔ_patterns)
                if n_match < 1 :
                  ipa_phoneme_sequence.extend([
                    'i', 'ʔ',
                    DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i+2]]
                  ])
                  i += 3; rule_found_flag = True
                else :
                  ipa_phoneme_sequence.extend([
                    'ə', 'ʔ',
                    DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i+2]]
                  ])
                  i += 3; rule_found_flag = True
        # AH|IH <consonant>
        if TWO_PHN_COND(i, rule_found_flag) and \
           arpabet_phoneme_sequence[i] in AH_IH_ARPABETS and \
           arpabet_phoneme_sequence[i+1] in CONSONANT_ARPABETS :
          if arpabet_phoneme_sequence[i] == "AH" :
            if arpabet_phoneme_sequence[i+1] == 'G' :
              # obs_flag = True
              aʔ_pattern = re.compile(r"TAG")
              if aʔ_pattern.search(grapheme) :
                ipa_phoneme_sequence.extend(['a', 'ʔ'])
                i += 2; rule_found_flag = True
              else :
                ipa_phoneme_sequence.extend(['ə', 'ʔ'])
                i += 2; rule_found_flag = True
            elif arpabet_phoneme_sequence[i+1] == 'K' :
              # obs_flag = True
              aʔ_pattern = re.compile(r"UCK")
              iʔ_pattern = re.compile(r"IC(K)?$")
              if aʔ_pattern.search(grapheme) :
                ipa_phoneme_sequence.extend(['a', 'ʔ'])
                i += 2; rule_found_flag = True
              elif iʔ_pattern.search(grapheme) :
                ipa_phoneme_sequence.extend(['i', 'ʔ'])
                i += 2; rule_found_flag = True
              else :
                ipa_phoneme_sequence.extend(['ə', 'ʔ'])
                i += 2; rule_found_flag = True
            elif arpabet_phoneme_sequence[i+1] == 'B' :
              # obs_flag = True
              if i+1 < len(arpabet_phoneme_sequence)-1 :
                ab_patterns = [
                  re.compile(r"OUB"),
                  re.compile(r"UB(?!$)")
                ]  
                n_match = sum(bool(ab_pattern.search(grapheme)) for ab_pattern in ab_patterns)
                if n_match < 1 :
                  ipa_phoneme_sequence.extend(['ə', 'b'])
                  i += 2; rule_found_flag = True
                else :
                  ipa_phoneme_sequence.extend(['a', 'b'])
                  i += 2; rule_found_flag = True
              else :
                ab_pattern = re.compile(r"UB$")
                if ab_pattern.search(grapheme) :
                  ipa_phoneme_sequence.extend(['a', 'b'])
                  i += 2; rule_found_flag = True
                else :
                  ipa_phoneme_sequence.extend(['ə', 'b'])
                  i += 2; rule_found_flag = True
            elif arpabet_phoneme_sequence[i+1] == "CH" :
              # obs_flag = True
              atʃ_patterns = [
                re.compile(r"UTCH"),
                re.compile(r"(?<!K)UCH")
              ]
              n_match = sum(bool(atʃ_pattern.search(grapheme)) for atʃ_pattern in atʃ_patterns)
              if n_match < 1 :
                ipa_phoneme_sequence.extend(['ə', "tʃ"])
                i += 2; rule_found_flag = True
              else :
                ipa_phoneme_sequence.extend(['a', "tʃ"])
                i += 2; rule_found_flag = True
            elif arpabet_phoneme_sequence[i+1] == 'D' :
              # obs_flag = True
              if i+1 < len(arpabet_phoneme_sequence)-1 :
                ad_pattern = re.compile(r"O?(?<!A)UD")
                if ad_pattern.search(grapheme) :
                  ipa_phoneme_sequence.extend(['a', 'd'])
                  i += 2; rule_found_flag = True
                else :
                  ipa_phoneme_sequence.extend(['ə', 'd'])
                  i += 2; rule_found_flag = True
              else :
                ad_pattern = re.compile(r"UDD?E?$")
                if ad_pattern.search(grapheme) :
                  ipa_phoneme_sequence.extend(['a', 'd'])
                  i += 2; rule_found_flag = True
                else :
                  ipa_phoneme_sequence.extend(['ə', 'd'])
                  i += 2; rule_found_flag = True
            elif arpabet_phoneme_sequence[i+1] == "DH" :
              obs_flag = True
              ad_pattern = re.compile(r"(O|U)TH")
              ed_pattern = re.compile(r"ATH")
              if ad_pattern.search(grapheme) :
                ipa_phoneme_sequence.extend(['a', 'd'])
                i += 2; rule_found_flag = True
              elif ed_pattern.search(grapheme) :
                ipa_phoneme_sequence.extend(['e', 'd'])
                i += 2; rule_found_flag = True
              else :
                ipa_phoneme_sequence.extend(['ə', 'd'])
                i += 2; rule_found_flag = True
          elif arpabet_phoneme_sequence[i] == "IH" :
            if arpabet_phoneme_sequence[i+1] == 'G' :
              # obs_flag = True
              eʔ_pattern = re.compile(r"NEG")
              if eʔ_pattern.search(grapheme) :
                ipa_phoneme_sequence.extend(['e', 'ʔ'])
                i += 2; rule_found_flag = True
              else :
                ipa_phoneme_sequence.extend(['i', 'ʔ'])
                i += 2; rule_found_flag = True
            elif arpabet_phoneme_sequence[i+1] == 'K' :
              # obs_flag = True
              eʔ_pattern = re.compile(r"ECK$")
              əʔ_pattern = re.compile(r"ECKE")
              if eʔ_pattern.search(grapheme) :
                ipa_phoneme_sequence.extend(['e', 'ʔ'])
                i += 2; rule_found_flag = True
              elif əʔ_pattern.search(grapheme) :
                ipa_phoneme_sequence.extend(['ə', 'ʔ'])
                i += 2; rule_found_flag = True
              else :
                ipa_phoneme_sequence.extend(['i', 'ʔ'])
                i += 2; rule_found_flag = True
            elif arpabet_phoneme_sequence[i+1] == 'B' :
              # obs_flag = True
              əb_pattern = re.compile(r"EB")
              if əb_pattern.search(grapheme) :
                ipa_phoneme_sequence.extend(['ə', 'b'])
                i += 2; rule_found_flag = True
              else :
                ipa_phoneme_sequence.extend(['i', 'b'])
                i += 2; rule_found_flag = True
            elif arpabet_phoneme_sequence[i+1] == "CH" :
              # obs_flag = True
              ətʃ_pattern = re.compile(r"(?<!R)EC")
              if ətʃ_pattern.search(grapheme) :
                ipa_phoneme_sequence.extend(['ə', "tʃ"])
                i += 2; rule_found_flag = True
              else :
                ipa_phoneme_sequence.extend(['i', "tʃ"])
                i += 2; rule_found_flag = True
            elif arpabet_phoneme_sequence[i+1] == 'D' :
              # obs_flag = True
              if i+1 < len(arpabet_phoneme_sequence)-1 :
                əd_patterns = [
                  re.compile(r"ZAD"),
                  re.compile(r"ED(?!$)")
                ]
                n_match = sum(bool(əd_pattern.search(grapheme)) for əd_pattern in əd_patterns)
                if n_match < 1 :
                  ipa_phoneme_sequence.extend(['i', 'd'])
                  i += 2; rule_found_flag = True
                else :
                  ipa_phoneme_sequence.extend(['ə', 'd'])
                  i += 2; rule_found_flag = True
              else :
                əd_pattern = re.compile(r"ED$")
                if əd_pattern.search(grapheme) :
                  ipa_phoneme_sequence.extend(['ə', 'd'])
                  i += 2; rule_found_flag = True
                else :
                  ipa_phoneme_sequence.extend(['i', 'd'])
                  i += 2; rule_found_flag = True
        # DH D => t d
        if TWO_PHN_COND(i, rule_found_flag) and \
           arpabet_phoneme_sequence[i] == "DH" and \
           arpabet_phoneme_sequence[i+1] == 'D' :
          # obs_flag = True
          ipa_phoneme_sequence.extend(['t', 'd'])
          i += 2; rule_found_flag = True
        # G|K HH|L|R|W|Y => g|k h|l|r|w|j
        if TWO_PHN_COND(i, rule_found_flag) and \
           arpabet_phoneme_sequence[i] in ['G', 'K'] and \
           arpabet_phoneme_sequence[i+1] in ["HH", 'L', 'R', 'W', 'Y'] :
          # obs_flag = True
          ipa_phoneme_sequence.extend([
            DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i]],
            DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i+1]],
          ])
          i += 2; rule_found_flag = True
        # <vocal> G|K<eos> => <corresp-vocal> ʔ
        # <r-ending> G|K<eos> => <corresp-r-ending> g|k
        #TODO: VOCAL_ARPABETS might need to be replaced with AH_IH_EXCLUDED_VOCAL_ARPABETS
        #TODO: more checking needed!! nested conditions are not mutually exclusive!! might have to resort to defaults
        if TWO_PHN_COND(i, rule_found_flag) and \
           arpabet_phoneme_sequence[i+1] in ['G', 'K'] and \
           i+1 == len(arpabet_phoneme_sequence)-1 :
          if arpabet_phoneme_sequence[i] in VOCAL_ARPABETS :
            # obs_flag = True
            ipa_phoneme_sequence.extend([
              DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i]],
              'ʔ'
            ])
            i += 2; rule_found_flag = True
          elif arpabet_phoneme_sequence[i] in R_ENDING_ARPABETS :
            # obs_flag = True
            ipa_phoneme_sequence.extend([
              DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i]],
              DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i+1]]
            ])
            i += 2; rule_found_flag = True
        # AH<eos> => a (A<eos> in grapheme)
        if ONE_PHN_COND(i, rule_found_flag) and \
           arpabet_phoneme_sequence[i] == "AH" and i==len(arpabet_phoneme_sequence)-1 and \
           grapheme.endswith('A') :
          # obs_flag = True
          ipa_phoneme_sequence.extend(['a'])
          i += 1; rule_found_flag = True
        # default
        if not rule_found_flag :
          ipa_phoneme_sequence.extend(DEFAULT_ARPABET_TO_IPA[arpabet_phoneme_sequence[i]].split())
          i += 1
      except Exception as e :
        print(f"An error occurred:")
        print(traceback.format_exc())
        sys.exit(1)
    if obs_flag :
      print(row[0])
      print(row[1], ipa_phoneme_sequence)
    # else :
    #   print(row[0], row[1], ipa_phoneme_sequence)