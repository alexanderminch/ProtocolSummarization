from transformers import pipeline

class TextSummarizer:
    def __init__(self, texts: dict, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        """
        Initializes the summarizer with classified text.

        Args:
            texts (dict): A dictionary where keys are class labels and values are strings of text.
            model_name (str): HuggingFace summarization model name.
        """
        self.texts = {k: v for k, v in texts.items() if v.strip()}  
        self.summarizer = pipeline("summarization", model=model_name)
        self.summaries = {}

    def summarize(self, max_words=512):
        """
        Summarizes the text for each class.

        Args:
            max_words (int): Maximum number of words to pass into the model per chunk.
        """
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

    def get_summary(self, label: str) -> str:
        """
        Returns the summary for a specific label.
        """
        return self.summaries.get(label, "[No summary available]")

    def get_all_summaries(self) -> dict:
        """
        Returns all label summaries.
        """
        return self.summaries
