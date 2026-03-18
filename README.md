# 📈 Metadata-Aware RAG with Self-Querying (SEC EDGAR)

An end-to-end, production-ready Retrieval-Augmented Generation (RAG) system built to handle complex financial documents (SEC EDGAR HTML filings). 

This project goes beyond standard semantic search by utilizing advanced retrieval techniques—specifically a **Self-Querying Retriever** and a custom **Multi-Query Retriever** pipeline—backed by a **Metadata-Aware Vector Database (Qdrant)**. This architecture achieves unmatched precision on computationally tricky queries (e.g., fetching data strictly from a specific company or year).

## 🧠 Advanced Retrieval Architecture

### 1. Self-Querying Retrieval
Standard RAG fails when a user asks: *"What were Apple's supply chain risks in 2022?"* The embedding model gets confused by the strict noun ("Apple") and integer ("2022") constraints. 
**The Solution:** We use a high-parameter LLM (`llama-3.3-70b-versatile`) to translate the natural language question into an exact structured Qdrant metadata payload filter *before* performing the vector search.
* **Benefit:** 100% precision on filtered queries. If the user asks for 2022 Apple data, the database physically ignores all other documents.

### 2. Multi-Query Retrieval
Semantic search relies on distance metrics, which means slight phrasing differences can cause the database to miss the best source document.
**The Solution:** Before querying, an LLM rewrites the user's base question from multiple distinct semantic perspectives. All variations are searched simultaneously, and the results are deduplicated.
* **Benefit:** Drastically higher recall rates. We configure this pipeline to extract the **Top K=2** most unique and relevant chunk proofs, guaranteeing a highly-focused, hallucination-free answer.

## 🚀 Key Features

*   **Self-Querying Retrieval:** Translates natural language user questions into exact structured Qdrant metadata payload filters.
*   **Multi-Query Retrieval:** Synthesizes multiple query variations to maximize context recall and deduplicates for the Top K=2 best proofs.
*   **Conversational Memory (History-Aware):** An intelligent pre-processing chain condenses follow-up questions containing pronouns (e.g., "What were the risks in 2023?") into standalone queries before searching.
*   **Robust HTML Ingestion:** Processes highly-complex SEC EDGAR filings cleanly by utilizing LangChain's `HTMLHeaderTextSplitter`. It preserves the semantic `<h>` tagging hierarchy rather than arbitrarily slicing paragraphs.
*   **Dynamic Multi-Dataset Parsing:** Automatically extracts metadata (`Company`, `Year`, `Form Type`) directly from HTML file names and content during the ingestion phase.
*   **Prompt Injection Guardrails:** Features a pre-execution security layer that intercepts adversarial prompts or "jailbreak" attempts before they hit the database.
*   **Automated RAG Evaluation:** Includes an evaluation script (`eval.py`) utilizing the **Ragas** framework to mathematically score Faithfulness, Answer Relevancy, and Context Precision against a golden dataset.

## 🛠️ Technology Stack

*   **Vector Database:** Qdrant (Local persistent disk)
*   **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Local, fast, free)
*   **LLM Engine:** Groq API (`llama-3.1-8b-instant` for general Q&A, `llama-3.3-70b-versatile` for complex Lisp generation)
*   **Frameworks:** LangChain, LangChain Classic
*   **App UI:** Streamlit
*   **Evaluation:** Ragas, Datasets

## 📂 Project Structure

```bash
Metadata-Aware-Rag/
├── app.py                  # Main Streamlit Chatbot application
├── ingest.py               # HTML ingestion, metadata extraction, and Qdrant population
├── eval.py                 # Ragas evaluation script
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (GROQ_API_KEY)
└── data/                   # Directory to hold raw HTML SEC filings
```

## ⚙️ How to Run Locally

### 1. Setup Environment
Clone the repository and set up a Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Keys
Copy the `.env.example` file to `.env` (or create a new `.env` file) and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Add Data & Ingest
1. Download SEC EDGAR filings in HTML format (e.g., Apple 10-K, Tesla 10-Q).
2. Place the `.html` files into the `data/` directory.
3. Run the ingestion pipeline to chunk the data and populate the Qdrant database:
```bash
python ingest.py
```

### 4. Start the Application
Boot up the Streamlit User Interface:
```bash
streamlit run app.py
```

## 📊 RAG Evaluation (Ragas)
To evaluate the system's performance metrics against the included "Golden Dataset":
```bash
python eval.py
```
This requires the Groq LLM to act as a "Judge" and will output a detailed `ragas_evaluation_report.csv` file scoring Hallucination instances, Answer Relevancy, and Context boundaries.
