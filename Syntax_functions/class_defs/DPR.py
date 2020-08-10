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

    ## Noun-Chunks: Are nouns and words that describe them in the text
    ##              Essentially returns subjects and objects of text. 
    def get_noun_chunks(self, sentence):
        result = []

        doc = self.model(sentence)

        for chunk in doc.noun_chunks:
            result.append((chunk.root.text, chunk.root.dep_,
                           chunk.root.head.text))
        return result

    ## Verbs: Action-words in sentences
    def get_verb(self, sentence):
        verbs = set()
        doc = self.model(sentence)

        for possible_subject in doc:
            if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
                verbs.add(possible_subject.head)
        if len(verbs) == 0:
            # print ("Could not find a verb.")
            return
        return verbs

    ## Returns all noun-chunks (subject and object) and verbs, in a dictionary. 
    def get_all(self, sentence):
        result = {}
        subj = []
        obj = []

        for ele in self.get_noun_chunks(sentence):
            if ele[1][1:] == 'subj':  # Grabbing Subject
                subj.append(ele[0])
            if ele[1][1:] == 'obj':  # Grabbing Object
                obj.append(ele[0])

        verbs = self.get_verb(sentence)

        if verbs is not None:
            result['Verbs'] = list(verbs)
        else:
            result['Verbs'] = []
        result['Subjects'] = subj
        result['Objects'] = obj

        return result

    """              Additional DPR ABILITY              """

    ### Returns pronouns as PERSON entities in sentences
    def pro_evaluate(self, sentence):
        None

    def pro_linking(self, sentence):
        None
        # d = gender.Detector()
        # print (d.get_gender("Sally"))

    """             Auxilliary functions                """

    ### Extracts sentences from a block of text, sentences serparated by a full
    ### stop. 
    def extract_sentences(self, paragraph):
        return paragraph.split('.')
