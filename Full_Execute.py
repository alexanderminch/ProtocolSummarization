import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Doc_Parsing import Pymu_Parse
from ZSE_Classifier import Classification

if __name__ == "__main__":
    parser = Pymu_Parse.catParse()
    parser.extract('Doc_Parsing/Prot_000 2.pdf')
    classifier = Classification.EmShot(parser.sorted_scrape)
    #classifier.cluster_graph()
