# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd
import numpy as np

# Takes input and output directories as arguments
parser=argparse.ArgumentParser()
parser.add_argument('--input', default=".", help='The file path of the unzipped DailyDialog dataset')
parser.add_argument('--output', default="./data", help='The file path of the output dataset')
args = parser.parse_args()
INPUT_PATH = args.input
OUTPUT_PATH = args.output

ISEAR_dataframe = pd.read_csv(INPUT_PATH+"/isear.csv", sep="|", error_bad_lines=False)
ISEAR_dataframe = pd.DataFrame({"dialogue": ISEAR_dataframe["SIT"], "emotion": ISEAR_dataframe["Field1"]})

fraction = 0.2

np.random.seed(seed=42)

test_indices = np.random.choice(ISEAR_dataframe.index, size=int(round(fraction*ISEAR_dataframe.shape[0])), replace=False)
train_indices = emobank.index.difference(test_indices)
dev_indices = np.random.choice(train_indices, size=int(round(fraction*len(train_indices))), replace=False)
train_indices = train_indices.difference(dev_indices)

ISEAR_train = ISEAR_dataframe.loc[train_indices,:]
ISEAR_dev = ISEAR_dataframe.loc[dev_indices,:]
ISEAR_test = ISEAR_dataframe.loc[test_indices,:]

# Make the output directory if it does not currently exist
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
    
ISEAR_train.to_csv(OUTPUT_PATH+"/train.tsv", sep='\t', encoding="utf-8")
ISEAR_dev.to_csv(OUTPUT_PATH+"/dev.tsv", sep='\t', encoding="utf-8")
ISEAR_test.to_csv(OUTPUT_PATH+"/test.tsv", sep='\t', encoding="utf-8")
