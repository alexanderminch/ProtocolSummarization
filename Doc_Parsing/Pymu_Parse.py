import fitz
#from difflib import get_close_matches

class catParse():

    def __init__(self):
        self.sorted_scrape = {}
        self.CATEGORIES = {
            "Study Objectives": ["purpose of the study", "study objectives"],
            "Study Endpoints": ["primary and secondary endpoints", "efficacy endpoints", "safety endpoints"],
            "Study Design": ["general design", "study schema", "randomisation and blinding"],
            "Study Population": ["population to be studied", "inclusion criteria", "exclusion criteria"],
            "Study Treatments": ["study treatments and dosage", "product preparation", "product accountability"],
            "Study Procedures": ["study schedule", "study visits", "withdrawal criteria"],
            "Study Monitoring": ["maintenance of randomisation", "source data"],
            "Study Methods": ["stopping rules", "methodology", "study methods"],
        }

    def match_to_category(self, heading):
        heading = heading.lower()
        for cat, keywords in self.CATEGORIES.items():
            if any(keyword in heading for keyword in keywords):
                return cat
        return "Unlabeled"
    
    # Heuristic-based filter for junk text, shortens classification time
    def filter_junk(self, text):
        if len(text.split()) < 10:
            return False
        alpha_chars = sum(c.isalpha() for c in text)
        total_chars = len(text)
        if total_chars == 0 or (alpha_chars / total_chars) < 0.3:
            return False
        punct_chars = sum(c in "!@#$%^&*()[]{};:,./<>?\|`~" for c in text)
        if (punct_chars / total_chars) > 0.3:
            return False
        return True

    def extract(self, pdf_path):
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()
        
        categorized_sections = {k: "" for k in self.CATEGORIES}
        categorized_sections["Unlabeled"] = ""

        for i, (lvl, title, page) in enumerate(toc):
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
                        categorized_sections[cat] += cleaned + " "

        self.sorted_scrape = categorized_sections

"""
if __name__ == "__main__":
    extract('Doc_Parsing\Prot_000 2.pdf')
"""
