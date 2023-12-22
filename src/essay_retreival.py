from openai import OpenAI
client = OpenAI()

#edited prompts file
path_file = '/Users/sanjayaharitsa/Downloads/projects/detect-aigen-text/detect-aigen-text/data/edited_prompts.txt'
new_path_file = '/Users/sanjayaharitsa/Downloads/projects/detect-aigen-text/detect-aigen-text/data/ai_essays.txt'
file = open(path_file, "r")
new_file = open(new_path_file, "a")
line = file.readlines()

for i in range(1000):
  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": line[i]}
    ]
  )
  new_file.write(response) #append this to train_essays.csv later?
  new_file.write("\n")

#close files
file.close()
new_file.close()