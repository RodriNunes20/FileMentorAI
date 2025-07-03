# FileMentorAI – Smart Document Knowledge Base
**Student Name:** *Rodrigo Nunes*  
**Date:** *03 July 2025*

---

## Features Implemented

### Day 2 Features (choose 3)

| ✅ | Feature | What it does |
|----|---------|--------------|
| ✔️ | **Document Manager** | Shows every uploaded file with preview & two-step delete (re-indexes Chroma after removal). |
| ✔️ | **Search History** | Keeps the last 10 Q&A pairs; sidebar pills let the user review past questions. |
| ✔️ | **Document Statistics** | Displays live metrics (doc count, total words) in the *Stats* tab. |

> **Why these three?** They cover the full search workflow: content management, recall of prior queries, and quantitative feedback.

### Day 3 Styling (choose 3)

| ✅ | Styling element | How it was applied |
|----|-----------------|--------------------|
| ✔️ | **Color Theme** | Global theme via `.streamlit/config.toml` & consistent `#2196F3` accent. |
| ✔️ | **Loading Animations** | Custom `st.spinner()` messages (“🔄 Crunching words…”, “🧠 Reasoning…”) around heavy calls. |
| ✔️ | **Icons & Visual Elements** | Tab icons, header emoji, translucent **card()** context-manager, and a fixed background image for brand feel. |

---

## How to Run

1. **Install dependencies**

   ```bash
   pip install streamlit chromadb transformers torch docling langchain_text_splitters

## Challenges and Solutions
1.Managing two-step delete without an official st.confirm() --> Stored a pending_{i} flag in st.session_state; second click checks the flag, then re-indexes and st.rerun().
2.Maintaining Chroma collection between Streamlit reruns --> Cached collection object in st.session_state; implemented reset_collection() helper to rebuild only when needed.
3.Background image reducing text contrast --> Created a card() context manager (rgba white 90 %) to wrap each tab’s body; optionally added a 35 % dark overlay layer.


## What I've learned
How to combine Streamlit UI primitives with a vector database
(Chroma) to build a lightweight retrieval-augmented chatbot.

The importance of state management (st.session_state) when
developing multi-step interactions in a stateless web environment.

Rapid UI polish with pure markdown/CSS (no external JS) can greatly
improve usability and grading clarity.
