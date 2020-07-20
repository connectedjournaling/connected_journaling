# Requirements:

pip install tensorflow <br>
pip install tensorflow-datasets

Python modules: <br>
Spacy <br>
sklearn <br>
numpy <br>
pandas <br>
nltk <br> 

python -m spacy download en_core_web_md --> run this in terminal
nltk.download('punkt') --> run this in python
  


# Data:
imdb data <br>
bbc-headline data


# Training:
Seem to only be able to process 10,000 observations @ a time, will need to worry about padding since they won't all be the same length
Use Google Collab.
If on local machine:
https://www.tensorflow.org/install/gpu
^ make sure these requirements are satisfied


