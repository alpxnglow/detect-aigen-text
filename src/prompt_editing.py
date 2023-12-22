#imports

#tacks
newPrompt =  ""
markerRemover = "Write a polished, final draft without paragraph markers "
ages = ["10 year-old", "high-school student", "college student", "adult", "old person"]
native = ["native English speaker. ", "non-native English speaker. "]
sourcesRemover = "Do not include a sources section."

##extract first line of "base_prompts"
path_file = '/Users/sanjayaharitsa/Downloads/projects/detect-aigen-text/detect-aigen-text/data/base_prompts.txt'
new_path_file = '/Users/sanjayaharitsa/Downloads/projects/detect-aigen-text/detect-aigen-text/data/edited_prompts.txt'
file = open(path_file, "r")
new_file = open(new_path_file, "a")
line = file.readlines()

#print(line[2])


#add tacks to prompt engineer
for x in range(len(native)):
   for y in range(len(ages)):
        for z in range(100):
            newPrompt = line[z].strip() + " " + markerRemover + "in the voice of a " + ages[y] + " who is a " + native[x] + sourcesRemover
            new_file.write(newPrompt)
            new_file.write("\n")

#close files
file.close()
new_file.close()
