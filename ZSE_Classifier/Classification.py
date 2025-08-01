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
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")

        # Affect Time/Accuracy of the Program:
        self.chunk_size = 150 # decides the size of text that is sent to sentence transformers, smaller = longer but higher classification accuracy, vice versa for larger
        self.cos_thresh = .7 # threshold for cos_sim
        self.cos_bot_thresh = .4 # if confidence of cos_sim doesn't meet this mark, discarded before sent to zero-shot
        # thus, chunks in range of (.4, .7) are sent to zero-shot
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
    
    def zs_classify(texts):
        pipe = pipeline(model="facebook/bart-large-mnli")
        result = pipe(texts,
                      candidate_labels=["Study Objectives", "Study Endpoints", "Study Design", 
                                        "Study Population", "Study Treatments", "Study Procedures",
                                        "Study Monitoring", "Study Methods"],)
    
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

    def compute_entropy(self, prob_dist):
        return -sum(p * math.log2(p) for p in prob_dist.values() if p > 0)

    def label(self):

        wb = load_workbook("Cosine_Similarity_Data.xlsx")
        ws = wb.active

        #keep
        words = self.texts["Unlabeled"].split()
        #chunks = [" ".join(words[i:i+150]) for i in range(0, int(len(words)), chunk_size)]

        chunk_sizes = list(range(100, 550, 50))
        confidences = []
        diffs = []
        ents = []
        for chunk_size in chunk_sizes:
            start_time = time.time()
            
            #keep
            chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
            for chunk in tqdm(chunks):
                similarity = self.cos_classify(chunk)

                confidences.append(similarity[1])
                diffs.append(similarity[2])
                ents.append(self.compute_entropy(similarity[3]))
            end_time = time.time()
            time_elapsed = round(end_time - start_time, 2)
            confidence_avg = round(sum(confidences) / len(confidences), 4) if confidences else 0
            diff_avg = round(sum(diffs) / len(diffs), 4) if diffs else 0
            avg_ent = sum(ents) / len(ents) if ents else 0
            ws.append([chunk_size, round(time_elapsed, 2), round(confidence_avg, 4), round(diff_avg, 4), round(avg_ent, 4)])

        wb.save("Cosine_Similarity_Data.xlsx")

    def cluster_graph(self, keep_top_percent=0.6):
        all_embeddings = []
        labels = []

        for cat in self.texts:
            if cat == "Unlabeled":
                continue

            # Split into chunks and remove empty ones
            chunks = self.texts[cat].split('.')
            chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
            if not chunks:
                continue

            # Embed the chunks
            embeddings = self.model.encode(chunks, convert_to_tensor=True)

            # Compute centroid and cosine similarities
            centroid = embeddings.mean(dim=0, keepdim=True)
            similarities = util.cos_sim(embeddings, centroid).squeeze(1)

            # Filter top N% most representative chunks
            k = int(len(similarities) * keep_top_percent)
            top_indices = similarities.argsort(descending=True)[:k]

            # Collect only top embeddings and their labels
            top_embeddings = embeddings[top_indices].cpu().numpy()
            all_embeddings.extend(top_embeddings)
            labels.extend([cat] * len(top_embeddings))

        # Convert all embeddings to NumPy array for t-SNE
        all_embeddings = numpy.array(all_embeddings)

        # Reduce dimensions using t-SNE
        reduced = TSNE(n_components=2, perplexity=30, n_iter=1000).fit_transform(all_embeddings)

        # Plotting
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