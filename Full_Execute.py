import streamlit as sl
import tempfile
import io, zipfile
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Doc_Parsing import Pymu_Parse
from ZSE_Classifier import Classification
from LLM_Summary import Summarizer

sl.set_page_config(page_title="Protocol Document Summarizer", layout="centered")
sl.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>Protocol Document Summarizer</h1>
    <p style='text-align: center; font-size: 18px;'>Upload a PDF, DOCX, or plain text document and get clean, categorized summaries using information from the entire document.</p>
    <hr style='margin-top: 20px; margin-bottom: 20px;'>
""", unsafe_allow_html=True)

file = sl.file_uploader("Drop Document Here", type=["pdf", "docx", "txt"])

if "parsed_text" not in sl.session_state:
    sl.session_state.parsed_text = None
if "classified" not in sl.session_state:
    sl.session_state.classified = None
if "summaries" not in sl.session_state:
    sl.session_state.summaries = None

if file and not sl.session_state.parsed_text:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.read())
        path = tmp.name
    with sl.spinner("Scraping Document"):
        parser = Pymu_Parse.catParse()
        parser.extract(path)
        sl.session_state.parsed_text = parser.sorted_scrape

if sl.session_state.parsed_text and not sl.session_state.classified:
    with sl.spinner("Processing Text"):
        classifier = Classification.EmShot(sl.session_state.parsed_text)
        classifier.getCatAvg()
        classifier.label()
        del classifier.texts["Unlabeled"]
        sl.session_state.classified = classifier.texts

if sl.session_state.classified and not sl.session_state.summaries:
    with sl.spinner("Summarizing..."):
        sum = Summarizer.TextSummarizer(sl.session_state.classified)
        sum.summarize()
        sl.session_state.summaries = sum.summaries

if sl.session_state.summaries:
    os.makedirs("Summaries", exist_ok=True)
    for label in sl.session_state.summaries:
        words = sl.session_state.summaries[label].split()
        wrap = '\n'.join([' '.join(words[i:i + 20]) for i in range(0, len(words), 20)])
        with open(os.path.join("Summaries", f"{label}.txt"), "w", encoding="utf-8") as f:
            f.write(f"== {label} ==\n{wrap}\n\n")

    sl.subheader("Summaries by Category")
    tabs = sl.tabs([f"Summary {i+1}" for i in range(len(sl.session_state.summaries))])
    for i, tab in enumerate(tabs):
        label = list(sl.session_state.summaries.keys())[i]
        with tab:
            sl.text_area("Summary", sl.session_state.summaries[label], height=300)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for label in sl.session_state.summaries:
            file_path = os.path.join("Summaries", f"{label}.txt")
            zip_file.write(file_path, arcname=f"{label}.txt")
    zip_buffer.seek(0)

    sl.download_button(
        label="Download Summaries",
        data=zip_buffer,
        file_name="summaries.zip",
        mime="application/zip"
    )

    # for label in sum.summaries:
    #     words = sum.summaries[label].split()
    #     wrap = '\n'.join([' '.join(words[i:i + 20]) for i in range(0, len(words), 20)])
    #     with open(os.path.join("Summaries", f"{label}.txt"), "w", encoding="utf-8") as f:
    #             f.write(f"== {label} ==\n{wrap}\n\n")
    # tabs = sl.tabs([f"Summary {i+1}" for i in range(len(sum.summaries))])
    # for i, tab in enumerate(tabs):
    #     with tab:
    #         sl.text_area("Summary", sum.summaries[i], height=300)