
from nltk.stem import WordNetLemmatizer
import re
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('stopwords')


class SimpleDataPreprocessor():

    def __init__(self):
        self.stop_words = ['that', 'a', 'to', 'of', 'which', 'and', 'while', 'in', 'for', 'those', 'their', 'these',
                          'this', 'but', 'howev', 'it', 'also', 'the', 'onli', 'have', 'one', 't', 's', 'v', 'd', 'at', 'has', 'what']
        self.stop_words = self.stop_words + list(set(stopwords.words('english')))
        self.stop_words = self.stop_words + list(set(stopwords.words('russian')))
        self.stop_words = set( self.stop_words )
        # for i in range(len(self.stop_words)):
        #    self.stop_words[i] = Stemmer.Stemmer("english").stemWord(self.stop_words[i])
        #self.translit = Transliterator()
        self.lemmatizer = WordNetLemmatizer()

        #self.uselessWords = []

    def preproc_doc_string(self, sample):
        doc_string = str(sample)
        doc_string = doc_string.lower()

        doc_string = re.sub('[^A-Za-zА-Яа-я\\s\t]+', ' ', doc_string)
        doc_string = doc_string.strip()

        doc_string = doc_string.split()
        for i in range(len(doc_string)):
            doc_string[i] = self.lemmatizer.lemmatize(doc_string[i])
        doc_string = ' '.join(doc_string)

        doc_string = doc_string.split()
        doc_string = ' '.join([i for i in doc_string if i not in self.stop_words])

        doc_string = re.sub('\n+', ' ', doc_string)
        doc_string = re.sub(' +', ' ', doc_string)
        doc_string = doc_string.strip()
        if doc_string == '' or doc_string == ' ':
            doc_string = '$$$STUB$$$'

        return doc_string

    def prerproc_docs(self, docs, n_jobs, remove_stub_strings=True):
        preprocessed_docs = Parallel(n_jobs, verbose=10)(delayed(self.preproc_doc_string)(doc) for doc in docs)
        #docs = np.hstack(docs)

        return preprocessed_docs

    def get_uniq_text_list(self, preproc_texts):
        uniq_texts = np.hstack([preproc_texts[:, 0], preproc_texts[:, 1]])
        uniq_texts = np.unique(uniq_texts)
        uniq_texts = list(sorted(list(uniq_texts)))
        return uniq_texts
