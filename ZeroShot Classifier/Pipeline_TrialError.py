from transformers import pipeline, AutoTokenizer
# from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from sentence_transformers import SentenceTransformer, util

# First attempt with powerful bert model, guessed correctly for study objectives with only 28% confidence,
# which prompted to me to search for different models, considering the paragraph felt fairly obvious.
# after further trials, determined what proved this model as the best was the difference in confidence
# from the first label to the second, most others were very close, but these were fairly differentiated

# BEST ZEROSHOT CLASSIFIER
pipe = pipeline(model="facebook/bart-large-mnli")
text = "Subjects are expected to participate in this study for a minimum of 53 weeks and up to 56 weeks, considering that the study will consist of a four week (maximum) screening period, and a 52 week open label phase."
result = pipe(text,
              candidate_labels=["Study Objectives", "Study Endpoints", "Study Design", 
                                "Study Population", "Study Treatments", "Study Procedures",
                                "Study Monitoring", "Study Methods"],)
print("Classified as: " + result['labels'][result['scores'].index(max(result['scores']))])

# Second attempt with a supposedly quicker model
# Not much quicker, not much more confident, still correct
"""
model = GLiClassModel.from_pretrained("knowledgator/gliclass-modern-base-v2.0-init")
tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-modern-base-v2.0-init", add_prefix_space=True)
pipeline = ZeroShotClassificationPipeline(model, tokenizer, classification_type='multi-label', device='cuda:0')

text = "The primary objective of the study is to evaluate the efficacy of Lanreotide ATG 120 mg in combination with TMZ in subjects with unresectable advanced neuroendocrine tumours of the lung or thymus (typical and atypical carcinoids according to the WHO 2004 criteria), as disease control rate (DCR) at 9 months, according to RECIST criteria v 1.1. "
labels = ["Study Objectives", "Study Endpoints", "Study Design", 
          "Study Population", "Study Treatments", "Study Procedures",
          "Study Monitoring", "Study Methods"]
result = pipeline(text, labels, threshold=0.5)[0] 
"""

# Third attempt 
# more confident, but in comparison to its confidence to the next highest, not great (.0001 difference vs roughly .15 for bert)
"""
classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
    tokenizer="MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
    device=-1  # use -1 for CPU
)

text = "This is a phase II, open label, single arm, prospective, multicenter, non-comparative, pilot study to evaluate the efficacy and safety of Lanreotide ATG 120 mg /28 days in combination with TMZ 250 mg/day for 5 consecutive days/28 days on Disease Control Rate (DCR), inadult subjects with a histologically documented unresectable advanced (locally or metastatic) well or moderately differentiated neuroendocrine tumor of the lung or thymus (typical and atypical carcinoids), according to the WHO 2004 criteria. The study consists of a screening period (maximum 4 weeks), followed by a 52 week open label phase.  "

candidate_labels = ["Study Objectives", "Study Endpoints", "Study Design",
                    "Study Population", "Study Treatments", "Study Procedures",
                    "Study Monitoring", "Study Methods"]

result = classifier(
    text,
    candidate_labels,
    hypothesis_template="This example is about {}",
    multi_label=True
)
"""


