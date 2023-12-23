import openai
import pandas as pd
import time

model_engine = "gpt-3.5-turbo" 

# Edited prompts file
path_file = 'data/edited_prompts.txt'
new_path_file = 'data/ai_essays.csv'
file = open(path_file, "r")
new_file = open(new_path_file, "a+")
line = file.readlines()

# Create a dummy dataframe
df = pd.DataFrame(columns=['essay', 'generated'])
num = 0

for i in range(1000):
  response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": line[i]}
    ]
  )
  df.loc[len(df.index)] = [response.choices[0].message.content, 1]
  print(df.loc[len(df.index)-1])
  df.to_csv(new_path_file, encoding='utf-8', index=False)
  num += 1
  if num == 5:
    time.sleep(60)
    num = 0

#close files
file.close()
new_file.close()
