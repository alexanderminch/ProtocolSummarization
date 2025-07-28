from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Doc_Parsing import Pymu_Parse

class EmShot():

    def __init__(self):
        parser = Pymu_Parse.catParse()
        self.texts = parser.extract('Doc_Parsing/Prot_000 2.pdf')
        self.cat_averages = {}
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        return self.model.encode(texts, convert_to_tensor=True)

    def classify(self, unkown, identifier):
        similarity = util.cos_sim(self.embed(unkown), self.embed(identifier))
    
    def getCatAvg(self):
        for cat in self.texts:
            words = self.texts[cat].split()
            chunks = [" ".join(words[i:i+150]) for i in range(0, len(words), 150)]
            embeddings = self.embed(chunks)  
            avg_embedding = torch.mean(embeddings, dim=0)  
            self.cat_averages[cat] = avg_embedding




