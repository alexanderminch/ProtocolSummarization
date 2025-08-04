import time
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Doc_Parsing import Pymu_Parse
from ZSE_Classifier import Classification
from LLM_Summary import Summarizer

if __name__ == "__main__":
    tot_time = 0
    parser = Pymu_Parse.catParse()
    print("Parsing Document...")
    start = time.time()
    parser.extract('Doc_Parsing/Prot_000 2.pdf')
    end = time.time()
    print(f"Parsing Complete in {(end - start)} Seconds")
    tot_time += end - start
    classifier = Classification.EmShot(parser.sorted_scrape)
    #classifier.cluster_graph()
    classifier.getCatAvg()
    print("Classifying Text...")
    start = time.time()
    classifier.label()
    end = time.time()
    tot_time += end - start
    print(f"Classifying Complete in {(end - start)/60} Minutes")
    del classifier.texts["Unlabeled"]
    print("Summarizing Text...")
    start = time.time()
    sum = Summarizer.TextSummarizer(classifier.texts)
    sum.summarize()
    end = time.time()
    tot_time += end - start
    print(f"Summary Complete in {(end - start)/60} Minutes")
    for label in sum.summaries:
        with open(os.path.join("Summaries", f"{label}.txt"), "w", encoding="utf-8") as f:
            f.write(f"== {label} ==\n{sum.summaries[label]}\n\n")
    print(f"All Summaries Written To File\nProgram Completed in: {tot_time}")
