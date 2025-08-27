import string
import itertools
from typing import Optional
import re
import nltk
import numpy as np
from text_recognizer.data.base_data_module import BaseDataModule
from nltk.corpus import brown

NLTK_DATA_DIRNAME = BaseDataModule.data_directory_path() / "downloaded" / "nltk"

class SentenceGen:
    """
    Generates Sentences
    """

    def __init__(self, max_length: Optional[int] = None):
        self.text = brown_text()
        self.word_start_inds = [0] + [_.start(0) + 1 for _ in re.finditer(" ", self.text)] # list of where in the string starts first letter of a word
        self.max_length = max_length

    def generate(self, max_length: Optional[int] = None) -> str:
        """
        Sample a string from text of the Brown corpus of length at least one word and at most max_length.
        Return:
            Returns a sentence
        """
        if max_length is None:
            max_length = self.max_length
        if max_length is None:
            raise ValueError("Must provide max_length to this method or when making an object.")
        
        for _ in range(10): #has max 10 tries to return right value
            try:
                # pick a random word start index
                first_indx = np.random.randint(0, len(self.word_start_inds) - 1)
                start_indx = self.word_start_inds[first_indx]

                # find possible end possitions
                end_ind_candidates = []
                for ind in range(first_indx + 1, len(self.word_start_inds)):
                    if self.word_start_inds[ind] - start_indx > max_length:
                        break
                    end_ind_candidates.append(self.word_start_inds[ind])

                # Randomly choose one of these end indx
                end_ind = np.random.choice(end_ind_candidates)
                # slice text from start index to end index
                sampled_text = self.text[start_indx:end_ind].strip()
                # substring is generated sentence
                return sampled_text
            except Exception:
                pass
        
        raise RuntimeError("Was not able to generate a valid string")



def brown_text():
    """
    Returns:
        Huge string of English words separated by 1 space.
    """
    sents = load_nltk_brown_corpus()
    text = " ".join(itertools.chain.from_iterable(sents)) # flattens the list of sentences into a single sequence of words
    text = text.translate({ord(c): None for c in string.punctuation}) # creates one giant string of all Brown corpus words
    text = re.sub("  +", " ", text) # removes double spaces

    return text

def load_nltk_brown_corpus():
    """
    Load the Brown corpus usinf NLTK library. Downloads it if not downloaded.
    Returns:
        List of sentences where each sentence is a list of words. ex: [["I", "am", "ok"], ["He", "is", "dog"]]
    """
    nltk.data.path.append(NLTK_DATA_DIRNAME)
    try:
        nltk.corpus.brown.sents()
    except LookupError:
        NLTK_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
        nltk.download("brown", download_dir=NLTK_DATA_DIRNAME)
    return nltk.corpus.brown.sents()