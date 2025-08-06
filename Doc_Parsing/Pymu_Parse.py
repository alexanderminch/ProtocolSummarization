import fitz
#from difflib import get_close_matches

class catParse():

    def __init__(self):
        self.sorted_scrape = {}
        self.CATEGORIES = {
            "Study Objectives": ["purpose of the study", "study objectives"],
            "Study Endpoints": ["primary and secondary endpoints", "efficacy endpoints", "safety endpoints"],
            "Study Design": ["general design", "study schema", "randomisation and blinding", "study population"],
            "Study Population": ["population to be studied", "inclusion criteria", "exclusion criteria", "study population"],
            "Study Treatments": ["study treatments and dosage", "product preparation", "product accountability"],
            "Study Procedures": ["study schedule", "study visits", "withdrawal criteria", "study procedures"],
            "Study Monitoring": ["maintenance of randomisation", "source data", "study monitoring"],
            "Study Methods": ["stopping rules", "methodology", "study methods"],
        }


    """
        Param: 
            heading (str) -> Titular text scraped from TOC
        Method:
            if a word in heading matches any word from the keywords in self.CATEGORIES, return (str) the category it matches to
            or Unlabeled if no match
        Returns:
            string

    """
    def match_to_category(self, heading):
        heading = heading.lower()
        for cat, keywords in self.CATEGORIES.items():
            if any(keyword in heading for keyword in keywords):
                return cat
        return "Unlabeled"
    
    """
        Heuristic-based filter for junk text, shortens classification time

        Param:
            text (str) -> line from pdf split on \n
        Method:
            if text is shorter than 10 words, less than 30% alphanumeric, more than 30% uppercase, or more than 30% 
            punctuation, return False. otherwise, True
        Returns:   
            boolean
    """
    def filter_junk(self, text):
        if len(text.split()) < 10:
            return False
        alpha_chars, total_chars = sum(c.isalpha() for c in text), len(text)
        upper = sum(1 for ch in text if ch.isupper())
        if total_chars == 0 or (alpha_chars / total_chars) < 0.3 or upper/total_chars > .3:
            return False
        punct_chars = sum(c in "!@#$%^&*()[]{;}:,./<>?\|`~" for c in text)
        if (punct_chars / total_chars) > 0.3:
            return False
        return True
    
    """
        Param:
            path (str) -> path to protocol document
        Method: 
            iterates through table of contents, matches the titles to one of the 9 categories, and adds their filtered/cleaned
            section text to that dictionary value
        Returns:
            null (edits self.sorted scrape in place)
    """
    def extract(self, path):
        doc = fitz.open(path)
        toc = doc.get_toc()
        
        self.sorted_scrape = {k: "" for k in self.CATEGORIES}
        self.sorted_scrape["Unlabeled"] = ""

        for i, (non, title, page) in enumerate(toc):
            title_lower = title.lower()
            start_page = page - 1
            end_page = toc[i + 1][2] - 1 if i + 1 < len(toc) else len(doc)
            cat = self.match_to_category(title_lower)
            for pno in range(start_page, end_page):
                raw = doc[pno].get_text()
                paragraphs = raw.split("\n\n")
                for par in paragraphs:
                    cleaned = ' '.join(par.split())  
                    if self.filter_junk(cleaned):
                        self.sorted_scrape[cat] += cleaned + " "

"""
if __name__ == "__main__":
    extract('Doc_Parsing\Prot_000 2.pdf')
"""
