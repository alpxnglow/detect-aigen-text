from openai import OpenAI
client = OpenAI()

# Prompt Engineering
markerRemover = "Write a polished, final draft without paragraph markers"
ages = ["10 year-old", "high-school student", "college student", "adult", "old person"]
native = ["native English speaker", "non-native English speaker"]
sourcesRemover = "Do not include a sources section"

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"}
  ]
)

print(response)