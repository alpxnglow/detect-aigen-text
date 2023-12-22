from openai import OpenAI
client = OpenAI()

#edited prompts file
path_file = '/Users/sanjayaharitsa/Downloads/projects/detect-aigen-text/detect-aigen-text/data/edited_prompts.txt'
file = open(path_file, "r")
line = file.readlines()

for i in range(100):
  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": line[i]}
    ]
  )
  print(response) #append this to a seperate text file later