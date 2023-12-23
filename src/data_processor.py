import pandas as pd

# Edited prompts file
human_generated_essays_file = 'data/train_essays.csv'
ai_generated_essays_file = 'data/ai_essays.csv'
hc3_file = 'data/all.jsonl'
all_essays_file = 'data/training_data.csv'

df_humans_data = pd.read_csv(human_generated_essays_file)
df_ai_data = pd.read_csv(ai_generated_essays_file)
df_hc3_data = pd.read_json('data/all.jsonl', lines=True)

df_humans_data.rename(columns = {'text':'essay'}, inplace = True)
df_humans_data = df_humans_data.drop(['id', 'prompt_id'], axis=1)

# Run through the code to get human answered an chatgpt answered text into separate rows
df_hc3_collated = pd.DataFrame(columns=['essay', 'generated'])
for i in range(df_hc3_data.count()[0]):
    df_hc3_collated.loc[len(df_hc3_collated.index)] = [df_hc3_data['chatgpt_answers'][i][0], 1]

    for j in range(3): # hard coding 3 since i notice most of the data have three responses from humans
        df_hc3_collated.loc[len(df_hc3_collated.index)] = [df_hc3_data['human_answers'][i][j], 0]

# df_hc3_data.rename(columns = {'text':'essay'}, inplace = True)
# df_hc3_data = df_humans_data.drop(['id', 'prompt_id'], axis=1)

df_all_data = pd.concat([df_humans_data, df_ai_data], ignore_index=True, sort=False)
df_all_data.to_csv(all_essays_file, encoding='utf-8', index=False)

# print(df_humans_data)
# print(df_humans_data.count())
# print(df_ai_data)
# print(df_ai_data.count())
# print(df_all_data)
# print(df_all_data.count())
