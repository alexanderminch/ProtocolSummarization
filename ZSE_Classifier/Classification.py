from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch
import numpy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class EmShot():

    def __init__(self, texts):
        self.texts = texts
        self.cat_averages = {}
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        if isinstance(texts, str): texts = [texts]
        return self.model.encode(texts, convert_to_tensor=True)

    def cos_classify(self, unkown):
        max_sim = ["", float('-inf')]
        for avg in self.cat_averages:
            sim = util.cos_sim(self.embed(unkown), self.cat_averages[avg])
            if sim > max_sim[1]:
                max_sim = [avg, sim]
        return max_sim[0]
    
    def zs_classify(texts):
        pipe = pipeline(model="facebook/bart-large-mnli")
        result = pipe(texts,
                candidate_labels=["Study Objectives", "Study Endpoints", "Study Design", 
                                    "Study Population", "Study Treatments", "Study Procedures",
                                    "Study Monitoring", "Study Methods"],)
    
    def getCatAvg(self):
        for cat in self.texts:
            if not self.texts[cat] or cat == "Unlabeled": continue
            words = self.texts[cat].split()
            chunks = [" ".join(words[i:i+150]) for i in range(0, len(words), 150)]
            embeddings = self.embed(chunks)  
            avg_embedding = torch.mean(embeddings, dim=0)  
            self.cat_averages[cat] = avg_embedding

    def label():
        pass

    def cluster_graph(self):
        all_embeddings = []
        labels = []
        for cat in self.texts:
            if cat == "Unlabeled": continue
            chunks = self.texts[cat].split('.')
            chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
            embeddings = self.model.encode(chunks, convert_to_tensor=True)

            all_embeddings.extend(embeddings.cpu().numpy())  
            labels.extend([cat] * len(chunks))

        all_embeddings = numpy.array(all_embeddings)
        reduced = TSNE(n_components=2, perplexity=30, n_iter=1000).fit_transform(all_embeddings)

        plt.figure(figsize=(12, 8))
        unique_labels = list(set(labels))
        colors = plt.cm.get_cmap("tab10", len(unique_labels))
        for i, label in enumerate(unique_labels):
            idxs = [j for j, lbl in enumerate(labels) if lbl == label]
            x = reduced[idxs, 0]
            y = reduced[idxs, 1]
            plt.scatter(x, y, color=colors(i), label=label, alpha=0.6)
        plt.title("Clustered Embeddings by Category")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()