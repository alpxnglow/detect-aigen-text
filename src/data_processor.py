import pandas as pd

# Edited prompts file
human_generated_essays_file = 'data/train_essays.csv'
ai_generated_essays_file = 'data/ai_essays.csv'
all_essays_file = 'data/training_data.csv'

df_humans_data = pd.read_csv(human_generated_essays_file)
df_ai_data = pd.read_csv(ai_generated_essays_file)

df_humans_data.rename(columns = {'text':'essay'}, inplace = True)
df_humans_data = df_humans_data.drop(['id', 'prompt_id'], axis=1)
df_all_data = pd.concat([df_humans_data, df_ai_data], ignore_index=True, sort=False)
df_all_data.to_csv(all_essays_file, encoding='utf-8', index=False)

<<<<<<< HEAD
# print(df_humans_data)
# print(df_humans_data.count())
# print(df_ai_data)
# print(df_ai_data.count())
# print(df_all_data)
# print(df_all_data.count())
=======
print(df_humans_data)
print(df_humans_data.count())
print(df_ai_data)
print(df_ai_data.count())
print(df_all_data)
print(df_all_data.count())
>>>>>>> 7fba661 (	modified:   data/edited_prompts.txt)
