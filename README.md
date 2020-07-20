# Requirements:

pip install tensorflow <br>
pip install tensorflow-datasets

Python modules: <br>
Spacy <br>
sklearn <br>
numpy <br>
pandas <br>
nltk <br> 

python -m spacy download en_core_web_md --> run this in terminal <br>
nltk.download('punkt') --> run this in python
  


# Data:
imdb data <br>
bbc-headline data


# Training:
Seem to only be able to process 10,000 observations @ a time, will need to worry about padding since they won't all be the same length
Use Google Collab.
If on local machine:
https://www.tensorflow.org/install/gpu
^ make sure these requirements are satisfied. CUDA 10.1, and cudNN libraries are installed
Make sure cuDNN libraries are copied and pasted to the correct place after downloading, and that the environment variables
are also pointed to the right place 


