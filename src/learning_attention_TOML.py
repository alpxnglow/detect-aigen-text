import numpy as np
import pandas as pd
import torch
import spacy

# Input and output files
GERMAN_ENGLIST_FILE = 'detect-aigen-text/data/translation_data/deu.txt'

# Read the file
df_deu = pd.read_csv(GERMAN_ENGLIST_FILE, sep='\t', names=["english", "german", "crap"])
print(df_deu.head(10))

