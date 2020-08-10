################################################################################
   
####   Description:  
   #   This script preprocess text files and prepares training data for the 
   #   NER model. 

####   Input: Text file
####   Output: Text file with annotated training data
####   Usage: python3 preprocess_NER.py HEALTH.txt well-being fitness
####          python3 preprocess_NER.py [ENTITY]  [WORD_SAMPLES]

################################################################################
import sys

## File with raw data
FILENAME = sys.argv[1]

## NER Category
CATEGORY = sys.argv[1][:-4]

## File with annotated text data
NEW_FILENAME = "annotated_" + CATEGORY + ".txt"

## Read raw data to a list (content)
with open(FILENAME) as f:
    content = f.readlines()

## Word samples in NER category
word_samples = sys.argv[1:]

## Open file for writing
new_file = open(NEW_FILENAME, "w")

for line in content: 
    entity_list = []

    for word in word_samples: 
        if word in line: 
            start_idx = line.index(word)
            end_idx = len(word) + start_idx 
            entity_list.append((start_idx, end_idx, CATEGORY))
    entity_dict = {'entities': entity_list}
    new_file.write('("' + line[:-1] + '", ' + str(entity_dict) + '),\n')

new_file.close()