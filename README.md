# Clinical Trial Protocol Document Summarizer

---

## Deliverable

Create a program to summarize CL Protocol docs (DOCX, PDF, or Plain Text) into 8 categories:
- **Study Objectives**
- **Study Endpoints**
- **Study Design**
- **Study Population**
- **Study Treatments**
- **Study Procedures**
- **Study Monitoring**
- **Study Methods**

For each topic, it compiles all information scattered throughout the report into one of the classifications. Then, using a LLM, summarizes and consolidates the amalgamation of texts into a concise and complete summary. 
---
## Execution

**Trial #1:** 

After some planning, the initial pipeline included 3 steps.
1. Parsing using some document universal python library, and breaking the entirety of the text into fragments.
2. Taking those fragments, and using a [zero-shot classifier](https://joeddav.github.io/blog/2020/05/29/ZSL.html) sort them into the 8 categories. 
3. Then finally, feeding the texts gathered into each classification and prompting a pre-trained LLM to summarize them. 

After some testing, the zero-shot classification proved not reliable enough, misclassifying greater than 1/3 of the time.

**Trial 2:** 

Following a few hours of research, a second pipeline was developed. I first interviewed a long-time Clinical 
Trial Manager on the practicality of the concept of the project. She said:

>"The sections of a protocol document are already well sectioned. Don't over-engineer it."

So I altered the classification process to a 3 step approach.
1. Classifying chunks based on their section header. 
2. Fringe or unlabled chunks would get [embedded](https://www.datacamp.com/blog/what-is-text-embedding-ai), then classified using cosine similarity compared to the average vector embedding for the already classified text. 
3. If any of those secondary chunks don't meet a confidence threshold, they are finally sent to a zero-shot classifier and labeled there.