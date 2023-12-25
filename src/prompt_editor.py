#edits
newPrompt =  ""
markerRemover = 'Write a formal and polishd final draft without paragraph markers (e.g., do not include "Title", "Introduction", "Paragraph 1", etc.) '
ages = ["10 year-old", "high-school student", "college student", "adult", "old person"]
native = ["native English speaker. ", "non-native English speaker. "]
sourcesRemover = 'Do not include a "Sources"/"Bibliography"/"Works Cited" section or anything of that sort. Make it seem human written. '
errorIntro = "Try to introduce minimal but appropriate spelling and grammatical errors for a "

##extract first line of "base_prompts"
path_file = 'data/base_prompts.txt'
new_path_file = 'data/edited_prompts.txt'
file = open(path_file, "r")
new_file = open(new_path_file, "a")
line = file.readlines()

#print(line[2])


#add tacks to prompt engineer
for x in range(100):
   for y in range(len(ages)):
        for z in range(len(native)):
            newPrompt = line[x].strip() + " " + markerRemover + "in the voice and style of a " + ages[y] + " who is a " + native[z] + sourcesRemover + errorIntro + ages[y] + "."
            new_file.write(newPrompt)
            new_file.write("\n")

#close files
file.close()
new_file.close()
