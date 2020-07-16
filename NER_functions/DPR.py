"""

    Description: This script presents a class that extracts dependencies in a 
    			 sentence.
    Inputs: Sentences
    Output: Dependencies
"""
import spacy 
from spacy.matcher import Matcher 
from spacy.tokens import Span 
from spacy.symbols import nsubj, VERB

import en_core_web_md

class DPR:
	def __init__(self, model):
		self.model = spacy.load(model)


	# Relation: Relationship between subject and objecs
	def get_relation(self, sentence):
		doc = self.model(sentence)

		# Matcher class object 
		matcher = Matcher(self.model.vocab)

		#define the pattern 
		pattern = [{'DEP':'ROOT'}, 
				{'DEP':'prep','OP':"?"},
				{'DEP':'agent','OP':"?"},  
				{'POS':'ADJ','OP':"?"}] 

		matcher.add("matching_1", None, pattern) 
		matches = matcher(doc)
		k = len(matches) - 1
		span = doc[matches[k][1]:matches[k][2]] 

		return(span.text)


	## Noun-Chunks: Are nouns and words that describe them in the text
	##				Essentially returns subjects and objects of text. 
	def get_noun_chunks(self, sentence):
		result = []
		doc = self.model(sentence)

		for chunk in doc.noun_chunks:
		    result.append((chunk.root.text, chunk.root.dep_,
		            chunk.root.head.text))
		return result


	## Verbs: Action-words in sentences
	def get_verbs(self, sentence):
		verbs = set()
		doc = self.model(sentence)

		for possible_subject in doc:
		    if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
		        verbs.add(possible_subject.head)
		if len(verbs) == 0: 
			print ("Could not find a verb.")
			return
		return verbs

