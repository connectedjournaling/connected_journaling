"""

    Description: This script presents a methodology to update entities with
                new word samples.
    Inputs: Requires previously trained NER examples for the model,
            New training data
    Output: Outputs the newly trained NER model
"""
from __future__ import unicode_literals, print_function
import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
#import gender_guesser.detector as gender
import pickle
import en_core_web_md


class NER:
    def __init__(self):
        self.TRAIN_DATA = []


    """                         DATA PREPROCESSING                           """
    ### To avoid the catastrophic forgetting problem, unpack data the NER
    ### model remembers.
    #       Input: training data (formatted list)
    #       Output: none
    def prepare_data(self, training_data):
        ## Unpickle file
        with open('NER_data.pkl', 'rb') as f:
            ## Store in list
            LIST = pickle.load(f)

        # Training data: Include here the new samples of words to update entities with.
        # format: ("One head. ", {'entities': [(0, 3, 'CARDINAL')]})
        #           In the format presened, 'One' belongs to the CARDINAL entity.
        TRAIN_DATA = training_data

        # Include this dataset from the pickle with the new TRAIN_DATA.
        self.TRAIN_DATA = LIST + TRAIN_DATA


    #       Description: updates data history with new entity samples
    #       Input: training data (formatted list)
    #       Output: none
    def update_data(self, training_data):
        ## Unpickle file - history of all training data
        with open('NER_data.pkl', 'rb') as f:
            ## Store in list
            LIST = pickle.load(f)

        # Include this dataset from the pickle with the new TRAIN_DATA.
        updated_LIST = LIST + training_data
        #Pickelize updated_LIST
        with open('NER_data.pkl', 'wb') as f:
            pickle.dump(updated_LIST, f)




    """              TRAIN EXISTING MODEL ON NEW WORD SAMPLES                """
    ### We update an NER model here, essentially updating the model with more
    ### word samples for an existing entity class.
    #       Input: model - Existing NER model (string)
    #       output_dir - Location of newly updated model
    #       n_iter - Number of training iterations
    #       Output: trained model at output_dir
    def update_entity(self, model=None, output_dir=None, n_iter=100):
        """Load the model, set up the pipeline and train the entity recognizer."""
        if model is not None:
            nlp = spacy.load(model)  # load existing spaCy model
            print("Loaded model '%s'" % model)
        else:
            nlp = spacy.blank("en")  # create blank Language class
            print("Created blank 'en' model")

        # create the built-in pipeline components and add them to the pipeline
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if "ner" not in nlp.pipe_names:
            ner = nlp.create_pipe("ner")
            nlp.add_pipe(ner, last=True)
        # otherwise, get it so we can add labels
        else:
            ner = nlp.get_pipe("ner")

        # add labels
        for _, annotations in self.TRAIN_DATA:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])

        # get names of other pipes to disable them during training
        pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
        # only train NER
        with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
            # show warnings for misaligned entity spans once
            warnings.filterwarnings("once", category=UserWarning, module='spacy')

            # reset and initialize the weights randomly – but only if we're
            # training a new model
            if model is None:
                nlp.begin_training()
            for itn in range(n_iter):
                random.shuffle(self.TRAIN_DATA)
                losses = {}
                # batch up the examples using spaCy's minibatch
                batches = minibatch(self.TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(
                        texts,  # batch of texts
                        annotations,  # batch of annotations
                        drop=0.5,  # dropout - make it harder to memorise data
                        losses=losses,
                    )
                print("Losses", losses)

        # test the trained model
        """print (" -----    TESTING THE TRAINED MODEL ON TRAINING DATA   -----")
        for text, _ in self.TRAIN_DATA:
            doc = nlp(text)
            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
            print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc]) 
        """

        # save model to output directory
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            nlp.to_disk(output_dir)
            print("Saved model to", output_dir)



    def add_entity(self, entity_name, model=None, output_dir=None, n_iter=30):
        """Set up the pipeline and entity recognizer, and train the new entity."""
        random.seed(0)
        if model is not None:
            nlp = spacy.load(model)  # load existing spaCy model
            print("Loaded model '%s'" % model)
        else:
            nlp = spacy.blank("en")  # create blank Language class
            print("Created blank 'en' model")

        # Add entity recognizer to model if it's not in the pipeline
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if "ner" not in nlp.pipe_names:
            ner = nlp.create_pipe("ner")
            nlp.add_pipe(ner)
        # otherwise, get it, so we can add labels to it
        else:
            ner = nlp.get_pipe("ner")

        ner.add_label(entity_name)  # add new entity label to entity recognizer
        # Adding extraneous labels shouldn't mess anything up
        ner.add_label("VEGETABLE")

        if model is None:
            optimizer = nlp.begin_training()
        else:
            optimizer = nlp.resume_training()
        move_names = list(ner.move_names)
        # get names of other pipes to disable them during training
        pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]


        # only train NER
        with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
            # show warnings for misaligned entity spans once
            warnings.filterwarnings("once", category=UserWarning, module='spacy')

            sizes = compounding(1.0, 4.0, 1.001)
            # batch up the examples using spaCy's minibatch
            for itn in range(n_iter):
                random.shuffle(self.TRAIN_DATA)
                batches = minibatch(self.TRAIN_DATA, size=sizes)
                losses = {}
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
                print("Losses", losses)


        # save model to output directory
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            #nlp.meta["name"] = new_model_name  # rename model
            nlp.to_disk(output_dir)
            print("Saved model to", output_dir)




    """              EVALUATE AN NER MODEL                """
    ### Evaluating performance of an NER model
    #       Input: model - Existing NER model (string)
    #       sentences - Sentences to test NER model on
    #       Output: NER results of sentences
    def evaluate(self, sentence, model=None):
        if model == None:
            print ("Please supply an NER model.")
            return
        if len(sentence) == 0:
            print ("Please provide more than 0 sentences.")
            return

        model = spacy.load(model)
        doc = model(sentence)
        
        return [(ent.text, ent.label_) for ent in doc.ents]


    




           
