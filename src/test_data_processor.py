import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET

HOME_DIR = str(Path.home())
HUMAN_ESSAYS_DIR = os.path.join(HOME_DIR, "Documents", "Misc", "data", "human_essays")
FILE_PATH = os.path.join(HUMAN_ESSAYS_DIR, "0002b.xml")

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
