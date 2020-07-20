import nltk
import pandas as pd
from pathlib import Path


def split_into_sentence(documents):
    sentences = []
    documents = pd.read_csv(documents)
    documents = documents['text']
    for document in documents:
        sent_text = nltk.sent_tokenize(document)
        for sentence in sent_text:
            sentences.append(sentence)

    return sentences