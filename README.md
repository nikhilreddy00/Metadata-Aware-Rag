# 📈 Metadata-Aware RAG with Self-Querying (SEC EDGAR)

An end-to-end, production-ready Retrieval-Augmented Generation (RAG) system built over financial documents (SEC EDGAR HTML filings). This project distinguishes itself from standard semantic search by utilizing a **Self-Querying Retriever** and a **Metadata-Aware Vector Database (Qdrant)** to achieve 100% precision on filtered queries (such as querying by specific company or year) while maintaining chat history.

## 🚀 Key Features

*   **Self-Querying Retrieval:** Uses Groq's `llama-3.3-70b-versatile` to translate natural language user questions into exact structured Qdrant metadata payload filters.
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
