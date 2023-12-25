import pandas as pd
import os
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET

HOME_DIR = str(Path.home())
MODEL_OUTPUT_DIR = os.path.join(HOME_DIR, "Documents", "Model", str(datetime.now().strftime("%m%d%Y%H%M%S")))
if not os.path.exists(MODEL_OUTPUT_DIR):
    os.mkdir(MODEL_OUTPUT_DIR)
MODEL_OUTPUT_LOCATION = os.path.join(MODEL_OUTPUT_DIR, "ai_human_essay_classifier.pth")


# Edited prompts file
human_generated_essays_file = 'data/human_essays.csv'
ai_generated_essays_file = 'data/ai_essays.csv'
all_essays_file = 'data/testing_data.csv'

df_humans_data = pd.read_csv(human_generated_essays_file)
df_ai_data = pd.read_csv(ai_generated_essays_file)

df_humans_data.rename(columns = {'text':'essay'}, inplace = True)
df_humans_data = df_humans_data.drop(['id', 'prompt_id'], axis=1)

df_all_data = pd.concat([df_humans_data, df_ai_data], ignore_index=True, sort=False)
df_all_data.to_csv(all_essays_file, encoding='utf-8', index=False)
