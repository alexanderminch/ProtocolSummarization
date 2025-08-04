from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch
import numpy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import time
from openpyxl import load_workbook
import torch.nn.functional as F
import math

class EmShot():

    def __init__(self, texts):
        self.texts = texts
        self.cat_averages = {}
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.pipe = pipeline(model="facebook/bart-large-mnli")

        # Affect Time/Accuracy of the Program:
        self.chunk_size = 150 # decides the size of text that is sent to sentence transformers, smaller = longer, vice versa for larger
        self.cos_thresh = .75 # threshold for cos_sim
        self.cos_bot_thresh = .6 # if confidence of cos_sim doesn't meet this mark, discarded before sent to zero-shot
        # thus, chunks in range of (.6, .75) are sent to zero-shot
        self.zs_thresh = .7 # confidence threshold for zero shot

    # embeds the chunk of text sent through as a tensor
    def embed(self, texts):
        if isinstance(texts, str): texts = [texts]
        return self.model.encode(texts, convert_to_tensor=True)

    # performs cosine similarity on the vector rep of the parameter text compared to the averages of all the pre classified texts
    # unkown -> list of strings
    # returns list [class chosen, confidence of classification, difference between first highest conf class and second, entropy of class probabilities]
    def cos_classify(self, unkown):
        similarities = {}
        for avg in self.cat_averages:
            sim = util.cos_sim(self.embed(unkown), self.cat_averages[avg])
            similarities[avg] = sim.item()  

        sort_sim = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        top_class, top_sim = sort_sim[0]
        second = sort_sim[1][1] if len(sort_sim) > 1 else 0.0

        sim_tensor = torch.tensor(list(similarities.values()))
        prob = F.softmax(sim_tensor, dim=0)
        full_probabilities = {cls: prob for cls, prob in zip(similarities.keys(), prob.tolist())}

        return [top_class, top_sim, top_sim - second, full_probabilities]
    
    def zs_classify(self, texts):
        print("Zero Shot Started...")
        result = self.pipe(texts,
                      candidate_labels=["Study Objectives", "Study Endpoints", "Study Design", 
                                        "Study Population", "Study Treatments", "Study Procedures",
                                        "Study Monitoring", "Study Methods"],)
        return result
    
    def getCatAvg(self, keep_top=0.7, chunk_size=150):
        for cat in self.texts:
            if not self.texts[cat] or cat == "Unlabeled": continue
            words = self.texts[cat].split()
            chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
            embeddings = self.embed(chunks)

            centroid = embeddings.mean(dim=0, keepdim=True)
            similarities = util.cos_sim(embeddings, centroid).squeeze(1)
            k = int(len(similarities) * keep_top)
            top_indices = similarities.argsort(descending=True)[:k]
            filtered_embeddings = embeddings[top_indices]
            self.cat_averages[cat] = torch.mean(filtered_embeddings, dim=0)

    def label(self):
        words = self.texts["Unlabeled"].split()
        chunks = [" ".join(words[i:i+150]) for i in range(0, int(len(words)), 300)]
        to_zs = []
        for chunk in tqdm(chunks, desc = "Embed"):
            similarity = self.cos_classify(chunk)
            if similarity[1] > .8: self.texts[similarity[0]] += chunk
            if similarity[1] > .7 and similarity[2] > .1: to_zs.append(chunk)
        guesses = self.zs_classify(to_zs)
        for chunk, guess in zip(to_zs, guesses):
            if not guess:
                print("Error: " + chunk)
            cl, best, diff = guess['labels'][0], guess['scores'][0], guess['scores'][0] - guess['scores'][1]
            if best > 0.75 or (best > 0.65 and diff > 0.1):
                self.texts[cl] += chunk
                

# -------- GRAPHING -----------

    def cluster_graph(self, keep_top_percent=0.6):
        all_embeddings = []
        labels = []
        for cat in self.texts:
            if cat == "Unlabeled": continue
            chunks = self.texts[cat].split('.')
            chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
            if not chunks:
                continue
            embeddings = self.model.encode(chunks, convert_to_tensor=True)
            centroid = embeddings.mean(dim=0, keepdim=True)
            similarities = util.cos_sim(embeddings, centroid).squeeze(1)
            k = int(len(similarities) * keep_top_percent)
            top_indices = similarities.argsort(descending=True)[:k]
            top_embeddings = embeddings[top_indices].cpu().numpy()
            all_embeddings.extend(top_embeddings)
            labels.extend([cat] * len(top_embeddings))

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
        plt.title(f"Top {int(keep_top_percent*100)}% Most Representative Chunks per Category")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()