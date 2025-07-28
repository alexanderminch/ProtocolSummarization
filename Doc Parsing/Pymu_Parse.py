import fitz
from difflib import get_close_matches

CATEGORIES = {
    "Study Objectives": ["purpose of the study", "study objectives"],
    "Study Endpoints": ["primary and secondary endpoints", "efficacy endpoints", "safety endpoints"],
    "Study Design": ["general design", "study schema", "randomisation and blinding"],
    "Study Population": ["population to be studied", "inclusion criteria", "exclusion criteria"],
    "Study Treatments": ["study treatments and dosage", "product preparation", "product accountability"],
    "Study Procedures": ["study schedule", "study visits", "withdrawal criteria"],
    "Study Monitoring": ["maintenance of randomisation", "source data"],
    "Study Methods": ["stopping rules", "methodology", "study methods"],
}

def match_to_category(heading):
    heading = heading.lower()
    for cat, keywords in CATEGORIES.items():
        if any(keyword in heading for keyword in keywords):
            return cat
    return "Unlabeled"

def extract(pdf_path):
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    
    categorized_sections = {k: "" for k in CATEGORIES}
    categorized_sections["Unlabeled"] = ""

    for i, (lvl, title, page) in enumerate(toc):
        title_lower = title.lower()
        start_page = page - 1
        end_page = toc[i + 1][2] - 1 if i + 1 < len(toc) else len(doc)

        cat = match_to_category(title_lower)

        for pno in range(start_page, end_page):
            categorized_sections[cat] += doc[pno].get_text()

    return categorized_sections

if __name__ == "__main__":
    extract('Doc Parsing\Prot_000 2.pdf')
