from datasets import load_dataset
import os
from torch.utils.data import DataLoader

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets/huggingface")
if not os.path.exists(DATASET_DIR) :
  os.mkdir(DATASET_DIR)

hf_dataset_sources = []
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_dataset_sources.txt")) as f :
  for line in f :
    hf_dataset_sources.append(line.strip())

for hf_dataset_source in hf_dataset_sources :
  paths = hf_dataset_source.split('/')
  for i, subpath in enumerate(paths) :
    if not os.path.exists(os.path.join(DATASET_DIR, '/'.join(paths[:i+1]))) :
      os.mkdir(os.path.join(DATASET_DIR, '/'.join(paths[:i+1])))
  for split, dataset in load_dataset(hf_dataset_source).items() :
    dataset.to_csv(os.path.join(DATASET_DIR, hf_dataset_source, f"{split}.csv"))
