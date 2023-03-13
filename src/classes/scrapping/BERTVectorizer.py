
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, TFBertModel
from sentence_transformers import SentenceTransformer

class BERTVectorizer():
    def __init__(self):
        
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        #self.model = TFBertModel.from_pretrained("bert-base-multilingual-cased")
        self.model = SentenceTransformer('bert-base-multilingual-cased')
        
        pass
    
    def vectorize_docs(self, url_contents):
        
        doc_vectors = self.model.encode( url_contents ) 
        
        return doc_vectors