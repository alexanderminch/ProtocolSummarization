from transformers import pipeline

class TextSummarizer:
    def __init__(self, texts: dict, llm: str = "sshleifer/distilbart-cnn-12-6"):
        """
        texts -> self.texts from EmShot class : dictionary with the 8 pre-defined labels + all their classified text
        """
        self.texts = {k: v for k, v in texts.items() if v.strip()}  
        self.summarizer = pipeline("summarization", model=llm)
        self.summaries = {}

    """
        Param:
            max_words (int): Maximum number of words to pass into the model per chunk.
        Method:
            splits classified texts into max_words length chunks, iterates through them summarizing each independently, then
            combining for a full category summary
    """
    def summarize(self, max_words=512):
        for label, full_text in self.texts.items():
            words = full_text.split()
            chunks = [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

            label_summary = []
            for chunk in chunks:
                try:
                    summary = self.summarizer(chunk, max_length=150, min_length=30, do_sample=False)
                    label_summary.append(summary[0]['summary_text'])
                except Exception as e:
                    print(f"Error summarizing chunk in label '{label}': {e}")

            self.summaries[label] = " ".join(label_summary)

    """
        Returns all label summaries.
    """
    def get_summaries(self) -> dict:
        return self.summaries
