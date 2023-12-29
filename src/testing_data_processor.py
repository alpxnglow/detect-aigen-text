import pandas as pd

# Input and output files
HUMAN_TEST_ESSAYS_FILE = 'data/human_test_essays.csv'
AI_TEST_ESSAYS_FILE = 'data/ai_test_essays.csv'
# Human sourced
AI_TEST_SOURCED_ESSAY_1_HUMAN_FILE = 'data/sourced_data/human/human_essay_1.csv'
AI_TEST_SOURCED_ESSAY_2_HUMAN_FILE = 'data/sourced_data/human/human_essay_2.csv'
AI_TEST_SOURCED_ESSAY_HEWLETT_HUMAN_FILE = 'data/sourced_data/human/human_essay_hewlett.csv'
AI_TEST_SOURCED_ESSAY_HUGG_HUMAN_FILE = 'data/sourced_data/human/human_essay_hugg.csv'
AI_TEST_SOURCED_POETRY_HUMAN_FILE = 'data/sourced_data/human/human_poetry.csv'
AI_TEST_SOURCED_PYCODE_HUMAN_FILE = 'data/sourced_data/human/human_code.csv'
AI_TEST_SOURCED_STORY_HUMAN_FILE = 'data/sourced_data/human/human_stories.csv'
# BARD sourced
AI_TEST_SOURCED_ESSAY_BARD_FILE = 'data/sourced_data/bard/BARD_essay.csv'
AI_TEST_SOURCED_POETRY_BARD_FILE = 'data/sourced_data/bard/BARD_poetry.csv'
AI_TEST_SOURCED_PYCODE_BARD_FILE = 'data/sourced_data/bard/BARD_pycode.csv'
AI_TEST_SOURCED_STORY_BARD_FILE = 'data/sourced_data/bard/BARD_story.csv'
## GPT sourced
AI_TEST_SOURCED_ESSAY_GPT_FILE = 'data/sourced_data/gpt/ChatGPT_essay.csv'
AI_TEST_SOURCED_POETRY_GPT_FILE = 'data/sourced_data/gpt/ChatGPT_poetry.csv'
AI_TEST_SOURCED_PYCODE_GPT_FILE = 'data/sourced_data/gpt/ChatGPT_pycode.csv'
AI_TEST_SOURCED_STORY_GPT_FILE = 'data/sourced_data/gpt/ChatGPT_story.csv'
# Output file
ALL_TEST_ESSAYS_FILE = 'data/testing_data.csv'

# DF setup
df_human_test_essays = pd.read_csv(HUMAN_TEST_ESSAYS_FILE)
df_ai_test_essays = pd.read_csv(AI_TEST_ESSAYS_FILE)

# Human
df_human_sourced_essay_1_human = pd.read_csv(AI_TEST_SOURCED_ESSAY_1_HUMAN_FILE)
df_human_sourced_essay_2_human = pd.read_csv(AI_TEST_SOURCED_ESSAY_2_HUMAN_FILE)
df_human_sourced_essay_hewlett_human = pd.read_csv(AI_TEST_SOURCED_ESSAY_HEWLETT_HUMAN_FILE)
df_human_sourced_essay_hugg_human = pd.read_csv(AI_TEST_SOURCED_ESSAY_HUGG_HUMAN_FILE)
df_human_sourced_poetry_human = pd.read_csv(AI_TEST_SOURCED_POETRY_HUMAN_FILE)
df_human_sourced_pycode_human = pd.read_csv(AI_TEST_SOURCED_PYCODE_HUMAN_FILE)
df_human_sourced_story_human = pd.read_csv(AI_TEST_SOURCED_STORY_HUMAN_FILE)

## BARD
df_ai_sourced_essay_bard = pd.read_csv(AI_TEST_SOURCED_ESSAY_BARD_FILE)
df_ai_sourced_poetry_bard = pd.read_csv(AI_TEST_SOURCED_POETRY_BARD_FILE)
df_ai_sourced_pycode_bard = pd.read_csv(AI_TEST_SOURCED_PYCODE_BARD_FILE)
df_ai_sourced_story_bard = pd.read_csv(AI_TEST_SOURCED_STORY_BARD_FILE)

## GPT
df_ai_sourced_essay_GPT = pd.read_csv(AI_TEST_SOURCED_ESSAY_GPT_FILE)
df_ai_sourced_poetry_GPT = pd.read_csv(AI_TEST_SOURCED_POETRY_GPT_FILE)
df_ai_sourced_pycode_GPT = pd.read_csv(AI_TEST_SOURCED_PYCODE_GPT_FILE)
df_ai_sourced_story_GPT = pd.read_csv(AI_TEST_SOURCED_STORY_GPT_FILE)

# Get the right columns in each df
## Human
df_human_sourced_essay_1_human.rename(columns = {'essays':'essay'}, inplace = True)
df_human_sourced_essay_1_human['generated'] = 0
df_human_sourced_essay_1_human.drop(df_human_sourced_essay_1_human.columns[[0]], axis=1, inplace=True)

df_human_sourced_essay_2_human.rename(columns = {'text':'essay'}, inplace = True)
df_human_sourced_essay_2_human['generated'] = 0
df_human_sourced_essay_2_human.drop(df_human_sourced_essay_2_human.columns[[0, 1]], axis=1, inplace=True)

df_human_sourced_essay_hewlett_human['generated'] = 0
df_human_sourced_essay_hewlett_human.drop(df_human_sourced_essay_hewlett_human.columns[[0]], axis=1, inplace=True)

df_human_sourced_essay_hugg_human.rename(columns = {'essays':'essay'}, inplace = True)
df_human_sourced_essay_hugg_human['generated'] = 0
df_human_sourced_essay_hugg_human.drop(df_human_sourced_essay_hugg_human.columns[[0, 1]], axis=1, inplace=True)

df_human_sourced_poetry_human.rename(columns = {'Poem':'essay'}, inplace = True)
df_human_sourced_poetry_human['generated'] = 0
df_human_sourced_poetry_human.drop(df_human_sourced_poetry_human.columns[[0, 1, 3, 4]], axis=1, inplace=True)

df_human_sourced_pycode_human.rename(columns = {'Code':'essay'}, inplace = True)
df_human_sourced_pycode_human['generated'] = 0
df_human_sourced_pycode_human.drop(df_human_sourced_pycode_human.columns[[0]], axis=1, inplace=True)

df_human_sourced_story_human.rename(columns = {'Chapter_text':'essay'}, inplace = True)
df_human_sourced_story_human['generated'] = 0
df_human_sourced_story_human.drop(df_human_sourced_story_human.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)

## BARD
df_ai_sourced_essay_bard.rename(columns = {'BARD':'essay'}, inplace = True)
df_ai_sourced_essay_bard['generated'] = 1
df_ai_sourced_essay_bard.drop(df_ai_sourced_essay_bard.columns[[0, 1]], axis=1, inplace=True)

df_ai_sourced_poetry_bard.rename(columns = {'BARD':'essay'}, inplace = True)
df_ai_sourced_poetry_bard['generated'] = 1
df_ai_sourced_poetry_bard.drop(df_ai_sourced_poetry_bard.columns[[0, 1]], axis=1, inplace=True)

df_ai_sourced_pycode_bard.rename(columns = {'BARD':'essay'}, inplace = True)
df_ai_sourced_pycode_bard['generated'] = 1
df_ai_sourced_pycode_bard.drop(df_ai_sourced_pycode_bard.columns[[0, 1]], axis=1, inplace=True)

df_ai_sourced_story_bard.rename(columns = {'BARD':'essay'}, inplace = True)
df_ai_sourced_story_bard['generated'] = 1
df_ai_sourced_story_bard.drop(df_ai_sourced_story_bard.columns[[0, 1]], axis=1, inplace=True)

## GPT
df_ai_sourced_essay_GPT.rename(columns = {'responses':'essay'}, inplace = True)
df_ai_sourced_essay_GPT['generated'] = 1
df_ai_sourced_essay_GPT.drop(df_ai_sourced_essay_GPT.columns[[0]], axis=1, inplace=True)

df_ai_sourced_poetry_GPT.rename(columns = {'responses':'essay'}, inplace = True)
df_ai_sourced_poetry_GPT['generated'] = 1
df_ai_sourced_poetry_GPT.drop(df_ai_sourced_poetry_GPT.columns[[0]], axis=1, inplace=True)

df_ai_sourced_pycode_GPT.rename(columns = {'responses':'essay'}, inplace = True)
df_ai_sourced_pycode_GPT['generated'] = 1
df_ai_sourced_pycode_GPT.drop(df_ai_sourced_pycode_GPT.columns[[0]], axis=1, inplace=True)

df_ai_sourced_story_GPT.rename(columns = {'Chapter_text':'essay'}, inplace = True)
df_ai_sourced_story_GPT['generated'] = 1
df_ai_sourced_story_GPT.drop(df_ai_sourced_story_GPT.columns[[0, 1]], axis=1, inplace=True)

# Run through the code to get human answered an chatgpt answered text into separate rows
df_all_data = pd.concat([df_human_test_essays, df_ai_test_essays, df_ai_sourced_essay_bard, df_ai_sourced_poetry_bard, df_ai_sourced_pycode_bard, df_ai_sourced_story_bard, df_ai_sourced_essay_GPT, df_ai_sourced_poetry_GPT, df_ai_sourced_pycode_GPT, df_ai_sourced_story_GPT, df_human_sourced_essay_1_human, df_human_sourced_essay_2_human, df_human_sourced_essay_hewlett_human, df_human_sourced_essay_hugg_human, df_human_sourced_poetry_human, df_human_sourced_pycode_human, df_human_sourced_story_human], ignore_index=True, sort=False)
df_all_data = df_all_data.dropna()
print(df_all_data)
df_all_data.to_csv(ALL_TEST_ESSAYS_FILE, encoding='utf-8', index=False)
