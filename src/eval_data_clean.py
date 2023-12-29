import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

HOME_DIR = str(Path.home())
HUMAN_ESSAYS_DIR = os.path.join(HOME_DIR, "Documents", "Misc", "data", "human_essays")
HUMAN_TEST_ESSAYS_FILE = 'data/human_test_essays.csv'

df_human_essays = pd.DataFrame(columns=['essay', 'generated'])

# iterate over files in
# that directory
for filename in os.listdir(HUMAN_ESSAYS_DIR):
    file_path = os.path.join(HUMAN_ESSAYS_DIR, filename)
    # checking if it is a file
    if os.path.isfile(file_path):
        file = open(file_path)
        contents = file.read()
        soup_original_xml = BeautifulSoup(contents, 'xml')
        sections_content = soup_original_xml.find_all('div1', {"type" : "section"})
        if len(sections_content) > 0:
            human_essay = ("\n".join(item for item in sections_content[0].get_text().split('\n') if item))
            df_human_essays.loc[len(df_human_essays.index)] = [human_essay, 0]

# Write the test essays to a file
df_human_essays.to_csv(HUMAN_TEST_ESSAYS_FILE, encoding='utf-8', index=False)
