import pandas as pd

# Input and output files
HUMAN_TEST_ESSAYS_FILE = 'data/human_test_essays.csv'
AI_TEST_ESSAYS_FILE = 'data/ai_test_essays.csv'
ALL_TEST_ESSAYS_FILE = 'data/testing_data.csv'

df_human_test_essays = pd.read_csv(HUMAN_TEST_ESSAYS_FILE)
df_ai_test_essays = pd.read_csv(AI_TEST_ESSAYS_FILE)

# Run through the code to get human answered an chatgpt answered text into separate rows
df_all_data = pd.concat([df_human_test_essays, df_ai_test_essays], ignore_index=True, sort=False)
df_all_data.to_csv(ALL_TEST_ESSAYS_FILE, encoding='utf-8', index=False)
