# Frequently Asked Questions (FAQ)

### What file formats are supported for resumes?
ResumeMatch supports **PDF (.pdf)** and **Word (.docx)** files. For DOCX files, the system also extracts text from tables, which are commonly used in resume templates.

### How is the Match Score calculated?
The score is a weighted average of five dimensions:
- **Skills (40%)**
- **Experience & Tenure (25%)**
- **Education (15%)**
- **Preferred Qualifications (10%)**
- **Overall Semantic Similarity (10%)**

### Why is my score lower than expected?
Check the "Missing" items in each section. Common reasons for a lower score include:
- **Terminology mismatch:** You might use a different name for a skill (though the Word2Vec expander tries to catch these).
- **Missing years:** The job might require "5+ years" and your resume only shows 3 based on detected dates.
- **Unstructured Resume:** If the text extraction fails to find dates or clear section headers, the experience score may be penalized.

### Where does the job data come from?
All job data is sourced from the official **Canadian Job Bank** via their CKAN open data portal. It is then enriched using the **NOC 2021** structure and **OaSIS** competency datasets.

### Can I run the analysis on my own custom job descriptions?
Yes. On the "Analyze" page, you can paste any job description text into the input field instead of selecting one from our database.

### How do "Rewrite Suggestions" work?
If you are missing a skill, the system looks for "semantically close" words already in your resume. For example, if you are missing "Kubernetes" but have "Docker" and "Cloud Orchestration," it will suggest highlighting those to show your related experience.

### Is my data stored?
The local development version of ResumeMatch does not store your uploaded resumes in any database. The text is processed in memory and cached temporarily in a volatile hash-map that clears when the server restarts.
